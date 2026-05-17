import os
import queue
import threading

import cv2
import numpy as np
import pyvista as pv
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image

from ros2_realsense.RealSenseCapture import RealsenseCapture


def _transform_stamped_to_matrix(msg: TransformStamped) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


# ---------------------------------------------------------------------------
# Reusable PyVista overlay base class
# ---------------------------------------------------------------------------

class PyVistaOverlay:
    """
    Off-screen PyVista renderer that composites a 3D scene onto a camera image.

    Coordinate convention: objects are placed in the OpenCV camera frame
    (X right, Y down, Z forward).  The VTK camera is set to match this
    convention, so no additional coordinate flip is needed in subclasses.

    Subclass and override _build_scene() to add arbitrary PyVista meshes.
    """

    def __init__(self, width: int, height: int, mtx: np.ndarray) -> None:
        self._width = width
        self._height = height

        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]

        # Off-screen plotter — black background so alpha compositing is trivial
        self._plotter = pv.Plotter(off_screen=True, window_size=[width, height])
        self._plotter.set_background([0, 0, 0])

        # Camera at origin, looking along +Z with Y down (OpenCV convention)
        cam = self._plotter.camera
        cam.position    = (0.0, 0.0, 0.0)
        cam.focal_point = (0.0, 0.0, 1.0)
        cam.view_up     = (0.0, -1.0, 0.0)

        # Vertical FOV derived from fy
        cam.view_angle = 2.0 * np.degrees(np.arctan(height / (2.0 * fy)))

        # Clipping planes — render objects 1 cm to 10 m away
        cam.clipping_range = (0.01, 10.0)

        # Handle non-square pixels (fx ≠ fy) via VTK's UseHorizontalViewAngle
        # VTK uses the vertical FOV by default (UseHorizontalViewAngle=False).
        # If fx != fy the horizontal angle will be set from fx separately.
        vtk_cam = self._plotter.renderer.GetActiveCamera()
        vtk_cam.SetUseHorizontalViewAngle(False)

        # Adjust for non-central principal point by shifting VTK's view frustum.
        # VTK window-centre coords are in [-1, 1] NDC per axis, where (0,0) is
        # the image centre.  A positive wx shifts the frustum right (principal
        # point moves left in image space), so we negate.
        wx = -(cx - width  / 2.0) / (width  / 2.0)
        wy =  (cy - height / 2.0) / (height / 2.0)   # Y already flipped by view_up
        vtk_cam.SetWindowCenter(wx, wy)

    def _build_scene(self, T_cam_object: np.ndarray) -> None:
        """Add meshes to self._plotter at the given camera-frame pose. Override in subclasses."""
        raise NotImplementedError

    def render(self, T_cam_object: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        Render the overlay at T_cam_object and alpha-composite onto background (BGR, H×W×3).
        Returns a BGR image of the same shape.
        """
        self._plotter.clear_actors()
        self._build_scene(T_cam_object)

        # screenshot returns RGBA (H×W×4) with transparent background
        rgba = self._plotter.screenshot(transparent_background=True, return_img=True)

        # RGBA → float BGR
        overlay_bgr = rgba[:, :, [2, 1, 0]].astype(np.float32)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0

        result = (1.0 - alpha) * background.astype(np.float32) + alpha * overlay_bgr
        return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Concrete overlay: coordinate frame (three RGB arrows)
# ---------------------------------------------------------------------------

class FrameOverlay(PyVistaOverlay):
    """
    Renders a coordinate frame as three arrows: X=red, Y=green, Z=blue.
    Arrows are pre-built in object-local space and transformed at render time.
    """

    # (direction, RGB color) for X, Y, Z axes
    _AXES = [
        ((1.0, 0.0, 0.0), [1.0, 0.0, 0.0]),
        ((0.0, 1.0, 0.0), [0.0, 1.0, 0.0]),
        ((0.0, 0.0, 1.0), [0.0, 0.0, 1.0]),
    ]

    def __init__(
        self,
        width: int,
        height: int,
        mtx: np.ndarray,
        axis_length: float = 0.1,
        shaft_radius: float = 0.03,   # fraction of axis_length
        tip_radius: float = 0.08,     # fraction of axis_length
        tip_length: float = 0.20,     # fraction of axis_length
    ) -> None:
        super().__init__(width, height, mtx)
        self._arrows = [
            (
                pv.Arrow(
                    start=(0.0, 0.0, 0.0),
                    direction=direction,
                    scale=axis_length,
                    shaft_radius=shaft_radius,
                    tip_radius=tip_radius,
                    tip_length=tip_length,
                ),
                color,
            )
            for direction, color in self._AXES
        ]

    def _build_scene(self, T_cam_object: np.ndarray) -> None:
        for mesh, color in self._arrows:
            m = mesh.copy()
            m.transform(T_cam_object, inplace=True)
            self._plotter.add_mesh(m, color=color, opacity=1.0)


# ---------------------------------------------------------------------------
# Concrete overlay: axis-aligned box (kept for future use)
# ---------------------------------------------------------------------------

class BoxOverlay(PyVistaOverlay):
    """Renders a solid or wireframe box centred at the object-frame origin."""

    def __init__(
        self,
        width: int,
        height: int,
        mtx: np.ndarray,
        box_size: list[float],
        color: list[float],
        opacity: float,
        style: str = "surface",
    ) -> None:
        super().__init__(width, height, mtx)
        sx, sy, sz = box_size
        self._mesh = pv.Box(
            bounds=[-sx / 2, sx / 2, -sy / 2, sy / 2, -sz / 2, sz / 2]
        )
        self._color = color
        self._opacity = opacity
        self._style = style

    def _build_scene(self, T_cam_object: np.ndarray) -> None:
        mesh = self._mesh.copy()
        mesh.transform(T_cam_object, inplace=True)
        self._plotter.add_mesh(
            mesh,
            color=self._color,
            opacity=self._opacity,
            style=self._style,
            line_width=3,
        )


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class OverlayObjectNode(Node):
    def __init__(self) -> None:
        super().__init__("overlay_object")

        self.declare_parameter("config_file", "")
        config_path = self.get_parameter("config_file").get_parameter_value().string_value
        if not config_path or not os.path.isfile(config_path):
            raise RuntimeError(f"config_file not found: '{config_path}'")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Load calibration files
        cal_dir = os.path.join(os.getcwd(), cfg["calibration"]["directory"])
        self._T_t_c = np.loadtxt(os.path.join(cal_dir, "T_tracker_camera.txt"))
        mtx = np.loadtxt(os.path.join(cal_dir, "camera_matrix.txt"))
        dist = np.loadtxt(os.path.join(cal_dir, "dist_coeffs.txt"))

        # Overlay
        obj_cfg = cfg["object"]
        self._overlay = FrameOverlay(
            width=1920,
            height=1080,
            mtx=mtx,
            axis_length=float(obj_cfg.get("axis_length", 0.1)),
            shaft_radius=float(obj_cfg.get("shaft_radius", 0.03)),
            tip_radius=float(obj_cfg.get("tip_radius", 0.08)),
            tip_length=float(obj_cfg.get("tip_length", 0.20)),
        )

        # Camera
        self._camera = RealsenseCapture(out_dir=os.path.join(cal_dir, "overlay_frames"))
        self._frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Shared tracker state
        self._lock = threading.Lock()
        self._T_world_object: np.ndarray | None = None
        self._T_world_tracker: np.ndarray | None = None

        self.create_subscription(
            TransformStamped,
            "/tracker_state_broadcaster/trackers/pose/MOZART_OBJECT",
            self._object_callback,
            10,
        )
        self.create_subscription(
            TransformStamped,
            "/tracker_state_broadcaster/trackers/pose/RealSense",
            self._tracker_callback,
            10,
        )

        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(Image, "~/image", 1)

        self.create_timer(1.0 / 30.0, self._update_callback)

        self.get_logger().info(
            f"Overlay node ready. Publishing to ~/image at 30 Hz.\n"
            f"  T^t_c loaded from {cal_dir}\n"
        )

    # ------------------------------------------------------------------

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

    def _object_callback(self, msg: TransformStamped) -> None:
        with self._lock:
            self._T_world_object = _transform_stamped_to_matrix(msg)

    def _tracker_callback(self, msg: TransformStamped) -> None:
        with self._lock:
            self._T_world_tracker = _transform_stamped_to_matrix(msg)

    def _update_callback(self) -> None:
        try:
            image, _ = self._frame_queue.get_nowait()
        except queue.Empty:
            return

        with self._lock:
            T_world_object  = self._T_world_object
            T_world_tracker = self._T_world_tracker

        now = self.get_clock().now().to_msg()

        if T_world_object is None or T_world_tracker is None:
            # No tracker data yet — publish raw frame
            msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
            msg.header.stamp = now
            msg.header.frame_id = "realsense_color_optical_frame"
            self._image_pub.publish(msg)
            return

        # T_world_camera = T_world_tracker @ T^t_c
        T_world_camera = T_world_tracker @ self._T_t_c

        # T_cam_object = inv(T_world_camera) @ T_world_object
        T_cam_object = np.linalg.inv(T_world_camera) @ T_world_object

        result = self._overlay.render(T_cam_object, image)

        msg = self._bridge.cv2_to_imgmsg(result, encoding="bgr8")
        msg.header.stamp = now
        msg.header.frame_id = "realsense_color_optical_frame"
        self._image_pub.publish(msg)

    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._stop_event.set()
        self._capture_thread.join(timeout=3.0)
        self._camera.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OverlayObjectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
