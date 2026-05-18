import os
import queue
import threading
from collections import deque

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from ros2_realsense.RealSenseCapture import RealsenseCapture


def _transform_stamped_to_matrix(msg: TransformStamped) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _rotation_angle(R: np.ndarray) -> float:
    """Angle (radians) of a rotation matrix via axis-angle."""
    return Rotation.from_matrix(R).magnitude()


class ArucoCalibrationNode(Node):
    def __init__(self):
        super().__init__("aruco_calibration")

        self.declare_parameter("config_file", "")
        config_path = self.get_parameter("config_file").get_parameter_value().string_value
        if not config_path or not os.path.isfile(config_path):
            raise RuntimeError(f"config_file parameter missing or not found: '{config_path}'")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # ArUco setup
        aruco_cfg = cfg["aruco"]
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, aruco_cfg["dictionary"])
        )
        self._marker_id = int(aruco_cfg["marker_id"])
        marker_length = float(aruco_cfg["marker_length"])
        ml = marker_length / 2.0
        self._obj_points = np.array(
            [[-ml, ml, 0], [ml, ml, 0], [ml, -ml, 0], [-ml, -ml, 0]], dtype=np.float32
        )
        detector_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        # Output setup
        out_cfg = cfg["output"]
        self._out_dir = os.path.join(os.getcwd(), out_cfg["directory"])
        os.makedirs(self._out_dir, exist_ok=True)
        self._min_samples = int(out_cfg.get("min_samples", 30))
        self._save_images = bool(out_cfg.get("save_images", True))

        # Auto-capture thresholds
        cap_cfg = cfg.get("capture", {})
        self._trans_threshold = float(cap_cfg.get("translation_threshold", 0.05))
        self._rot_threshold = float(np.deg2rad(cap_cfg.get("rotation_threshold_deg", 10.0)))
        self._static_duration = float(cap_cfg.get("static_duration", 1.0))
        self._static_trans_tol = float(cap_cfg.get("static_translation_tol", 0.003))
        self._static_rot_tol = float(np.deg2rad(cap_cfg.get("static_rotation_tol_deg", 0.5)))

        # Camera init
        self._camera = RealsenseCapture(out_dir=self._out_dir, width=1920, height=1080, fps=6)

        # Shared state
        self._lock = threading.Lock()
        # Pose history: deque of (stamp_sec: float, T: np.ndarray)
        self._pose_history: deque[tuple[float, np.ndarray]] = deque()
        self._latest_tracker: np.ndarray | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=5)

        # Calibration data
        self._tracker_poses: list[np.ndarray] = []
        self._aruco_poses: list[np.ndarray] = []
        self._last_captured_pose: np.ndarray | None = None
        self._T_t_c: np.ndarray | None = None          # current best estimate of T^t_c
        self._latest_T_ci_o: np.ndarray | None = None  # latest live ArUco detection
        self._done = False

        # RViz publishers
        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(Image, "~/image", 1)
        self._camera_info_pub = self.create_publisher(CameraInfo, "~/camera_info", 1)
        self._camera_info_msg = self._build_camera_info_msg()
        self._tf_broadcaster = TransformBroadcaster(self)

        # Tracker subscriber
        self._tracker_sub = self.create_subscription(
            TransformStamped,
            "/tracker_state_broadcaster/trackers/pose/RealSense",
            self._tracker_callback,
            10,
        )

        # Preview + auto-capture timer (10 Hz)
        self._preview_timer = self.create_timer(0.1, self._preview_callback)

        # Background camera capture thread
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.get_logger().info(
            f"ArUco calibration ready — auto-capture enabled.\n"
            f"  Marker: dict={aruco_cfg['dictionary']} id={self._marker_id} "
            f"length={marker_length}m\n"
            f"  Thresholds: move >{self._trans_threshold*100:.0f}cm or "
            f">{np.rad2deg(self._rot_threshold):.0f}° from last pose, "
            f"static for ≥{self._static_duration}s\n"
            f"  Goal: {self._min_samples} pairs → {self._out_dir}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_camera_info_msg(self) -> CameraInfo:
        intr = self._camera.intr
        msg = CameraInfo()
        msg.header.frame_id = "realsense_color_optical_frame"
        msg.width = intr.width
        msg.height = intr.height
        msg.distortion_model = "plumb_bob"
        msg.d = list(self._camera.dist)
        mtx = self._camera.mtx
        msg.k = [
            mtx[0, 0], 0.0, mtx[0, 2],
            0.0, mtx[1, 1], mtx[1, 2],
            0.0, 0.0, 1.0,
        ]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [
            mtx[0, 0], 0.0, mtx[0, 2], 0.0,
            0.0, mtx[1, 1], mtx[1, 2], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    @staticmethod
    def _matrix_to_tf(T: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(T[0, 3])
        msg.transform.translation.y = float(T[1, 3])
        msg.transform.translation.z = float(T[2, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        return msg

    def _check_static(self, history_snapshot: list[tuple[float, np.ndarray]]) -> bool:
        """True if the pose has not moved beyond tolerance for the last static_duration seconds."""
        if not history_snapshot:
            return False
        latest_stamp = history_snapshot[-1][0]
        window_start = latest_stamp - self._static_duration
        window = [(t, T) for t, T in history_snapshot if t >= window_start]
        # Need entries spanning the full window
        if not window or window[0][0] > window_start + 0.1:
            return False
        ref_T = window[0][1]
        for _, T in window[1:]:
            if np.linalg.norm(T[:3, 3] - ref_T[:3, 3]) > self._static_trans_tol:
                return False
            R_rel = ref_T[:3, :3].T @ T[:3, :3]
            if _rotation_angle(R_rel) > self._static_rot_tol:
                return False
        return True

    def _check_different_enough(self, T_current: np.ndarray) -> bool:
        """True if current pose differs enough from the last captured pose."""
        if self._last_captured_pose is None:
            return True
        delta_t = np.linalg.norm(T_current[:3, 3] - self._last_captured_pose[:3, 3])
        R_rel = self._last_captured_pose[:3, :3].T @ T_current[:3, :3]
        delta_r = _rotation_angle(R_rel)
        return delta_t > self._trans_threshold or delta_r > self._rot_threshold

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _tracker_callback(self, msg: TransformStamped) -> None:
        T = _transform_stamped_to_matrix(msg)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._latest_tracker = T
            self._pose_history.append((stamp, T))
            # Keep only the last (static_duration + 1) seconds
            cutoff = stamp - (self._static_duration + 1.0)
            while self._pose_history and self._pose_history[0][0] < cutoff:
                self._pose_history.popleft()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            image, ts = self._camera.capture()
            if image is None:
                continue
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put((image, ts))

    def _preview_callback(self) -> None:
        if self._done:
            return

        try:
            image, ts = self._frame_queue.get_nowait()
        except queue.Empty:
            return

        with self._lock:
            T_t = self._latest_tracker.copy() if self._latest_tracker is not None else None
            history_snapshot = list(self._pose_history)

        # ArUco detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        marker_detected = ids is not None and self._marker_id in ids.flatten()

        rvec = tvec = None
        if marker_detected:
            idx = int(np.where(ids.flatten() == self._marker_id)[0][0])
            img_pts = corners[idx][0].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_points, img_pts, self._camera.mtx, self._camera.dist
            )
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                T_ci_o = np.eye(4)
                T_ci_o[:3, :3] = R
                T_ci_o[:3, 3] = tvec.flatten()
                self._latest_T_ci_o = T_ci_o
            else:
                marker_detected = False

        # Evaluate auto-capture conditions
        is_static = T_t is not None and self._check_static(history_snapshot)
        is_different = T_t is not None and self._check_different_enough(T_t)
        will_capture = marker_detected and is_static and is_different
        n = len(self._tracker_poses)

        # Build annotated image
        annotated = image.copy()
        if marker_detected:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            cv2.drawFrameAxes(
                annotated, self._camera.mtx, self._camera.dist,
                rvec, tvec, self._obj_points[1, 0],
            )

        # Status overlay
        if will_capture:
            border_color = (0, 255, 0)
            status = "CAPTURING"
        elif is_static and not is_different:
            border_color = (0, 200, 255)
            status = "STATIC — move to new pose"
        elif not is_static:
            border_color = (0, 165, 255)
            status = "MOVING"
        else:
            border_color = (0, 0, 255)
            status = "Marker not detected"

        cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1),
                      border_color, 6)
        cv2.putText(annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Pairs: {n}/{self._min_samples}",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Publish image to RViz
        now = self.get_clock().now().to_msg()
        img_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = "realsense_color_optical_frame"
        self._image_pub.publish(img_msg)
        self._camera_info_msg.header.stamp = now
        self._camera_info_pub.publish(self._camera_info_msg)

        # Broadcast TF transforms derived from the current best T^t_c estimate
        if T_t is not None and self._T_t_c is not None:
            # Camera pose in world: T_world_camera = T_t @ T^t_c
            T_cam_world = T_t @ self._T_t_c
            self._tf_broadcaster.sendTransform(
                self._matrix_to_tf(T_cam_world, "world", "realsense_color_optical_frame", now)
            )
            # Marker pose in world (live): T_world_marker = T_t @ T^t_c @ T^ci_o
            if self._latest_T_ci_o is not None:
                T_marker_world = T_cam_world @ self._latest_T_ci_o
                self._tf_broadcaster.sendTransform(
                    self._matrix_to_tf(T_marker_world, "world", "aruco_marker", now)
                )

        # Auto-capture
        if will_capture:
            self._store_pair(image, ts, corners, ids, rvec, tvec, T_t)

    def _store_pair(
        self,
        image: np.ndarray,
        ts: float,
        corners,
        ids,
        rvec: np.ndarray,
        tvec: np.ndarray,
        T_t: np.ndarray,
    ) -> None:
        R, _ = cv2.Rodrigues(rvec)
        T_ci_o = np.eye(4)
        T_ci_o[:3, :3] = R
        T_ci_o[:3, 3] = tvec.flatten()

        self._tracker_poses.append(T_t)
        self._aruco_poses.append(T_ci_o)
        self._last_captured_pose = T_t.copy()
        n = len(self._tracker_poses)

        if self._save_images:
            annotated = image.copy()
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            cv2.drawFrameAxes(
                annotated, self._camera.mtx, self._camera.dist,
                rvec, tvec, self._obj_points[1, 0],
            )
            cv2.imwrite(os.path.join(self._out_dir, f"pair_{n:04d}_{ts:.3f}.png"), annotated)

        self.get_logger().info(f"Auto-captured pair {n}/{self._min_samples}")

        if n >= 3:
            self._solve_intermediate(n)

        if n >= self._min_samples:
            self._done = True
            self._solve_and_save()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _run_hand_eye(self) -> np.ndarray:
        """Run calibrateHandEye on all collected pairs and return T^t_c (4×4)."""
        R_t = [T[:3, :3] for T in self._tracker_poses]
        t_t = [T[:3, 3:4] for T in self._tracker_poses]
        R_c = [T[:3, :3] for T in self._aruco_poses]
        t_c = [T[:3, 3:4] for T in self._aruco_poses]
        R_x, t_x = cv2.calibrateHandEye(
            R_t, t_t, R_c, t_c, method=cv2.CALIB_HAND_EYE_TSAI
        )
        T = np.eye(4)
        T[:3, :3] = R_x
        T[:3, 3] = t_x.flatten()
        return T

    def _solve_intermediate(self, n: int) -> None:
        """Recompute T^t_c from all pairs so far; update TF broadcast state."""
        try:
            T_t_c = self._run_hand_eye()
        except Exception as e:
            self.get_logger().warn(f"Intermediate solve failed ({n} pairs): {e}")
            return
        self._T_t_c = T_t_c
        t = T_t_c[:3, 3]
        euler = Rotation.from_matrix(T_t_c[:3, :3]).as_euler("xyz", degrees=True)
        self.get_logger().info(
            f"[{n} pairs] T^t_c  t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]  "
            f"rpy=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°"
        )

    def _solve_and_save(self) -> None:
        self.get_logger().info("Solving final AX=XB hand-eye calibration...")

        T_t_c = self._run_hand_eye()
        self._T_t_c = T_t_c
        R_x = T_t_c[:3, :3]
        t_x = T_t_c[:3, 3:4]

        self.get_logger().info(f"T^t_c (camera in tracker frame):\n{np.round(T_t_c, 6)}")

        txt_path = os.path.join(self._out_dir, "T_tracker_camera.txt")
        np.savetxt(txt_path, T_t_c, fmt="%.8f")

        rot = Rotation.from_matrix(R_x)
        quat = rot.as_quat()
        euler_deg = rot.as_euler("xyz", degrees=True)
        yaml_path = os.path.join(self._out_dir, "T_tracker_camera.yaml")
        result = {
            "T_tracker_camera": {
                "translation": {"x": float(t_x[0]), "y": float(t_x[1]), "z": float(t_x[2])},
                "quaternion": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
                "euler_xyz_deg": {
                    "r": float(euler_deg[0]),
                    "p": float(euler_deg[1]),
                    "y": float(euler_deg[2]),
                },
            }
        }
        with open(yaml_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False)

        self.get_logger().info(f"Saved T^t_c → {txt_path}  and  {yaml_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._stop_event.set()
        self._capture_thread.join(timeout=3.0)
        self._camera.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
