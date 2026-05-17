import os
import cv2
import time
import numpy as np
import multiprocessing as mp
import pickle as pkl

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from RealSenseCapture import RealsenseCapture


def object_pose(rvec, tvec, O_T_C):
    T = np.eye(4)
    R = cv2.Rodrigues(rvec)[0]
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    return O_T_C @ T


class ArucoPose:
    def __init__(self, mtx, dist, detector, marker_length):
        self.mtx = mtx
        self.dist = dist
        self.detector = detector
        self.marker_length = marker_length

    def get_pose(self, color_image):
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray_image)

        # get pose if marker detected
        if ids is not None:
            # Define the 3D coordinates of the marker corners in the marker's coordinate system
            obj_points = np.array(
                [
                    [-self.marker_length / 2, self.marker_length / 2, 0],
                    [self.marker_length / 2, self.marker_length / 2, 0],
                    [self.marker_length / 2, -self.marker_length / 2, 0],
                    [-self.marker_length / 2, -self.marker_length / 2, 0],
                ],
                dtype=np.float32,
            )

            corner = corners[0]

            img_points = corner[0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, self.mtx, self.dist
            )
            return corners, ids, success, (rvec, tvec)

        else:
            return corners, ids, False, (None, None)


class PoseEstimator(RealsenseCapture):
    def __init__(self, out_dir, marker_length):
        # init camera
        super().__init__(out_dir=out_dir)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.aruco_pose = ArucoPose(self.mtx, self.dist, detector, marker_length)
       
        success = False
        counter = 0
        while not success:
            color_image, _ = self.capture(0.0,0, False, False)

            corners, ids, success, (rvec, tvec) = self.aruco_pose.get_pose(color_image)
            if success:
                self.tvec = tvec
                self.rvec = rvec
            else:
                print(f"Waiting for marker detection to initialize pose filter ", end="\r")
                time.sleep(0.1)
            counter += 1
            if counter > 10:
                self.tvec = np.array([[0.5], [0.0], [0.5]])
                break

       

    def estimate(self, time, idx):
        # this is a blocking call - waits for the next set of frames
        color_image, capture_time = self.capture(time,idx,show_frame=False, save_frame=False)

        corners, ids, success, (rvec, tvec) = self.aruco_pose.get_pose(color_image)

        # if ids is not None:
        #     cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        #     cv2.drawFrameAxes(color_image, self.mtx, self.dist, rvec, tvec, self.aruco_pose.marker_length)

        filename = f"image_{idx:04d}_{time:.3f}.png"
        image_filename = os.path.join(self.out_dir, filename)
        cv2.imwrite(image_filename, color_image)

        return color_image, corners, ids, success, (rvec, tvec), capture_time


def start_pose_estimation_loop(stop_event, clock, conn, out_dir, marker_length):
    pose_estimator = PoseEstimator(out_dir, marker_length)
    counter = 0
    while True:
        if stop_event.is_set():
            print("Stopping pose estimation loop...")
            break

        _, corners, _, success, (rvec, tvec), capture_time = pose_estimator.estimate(clock.value, counter)
        counter += 1

        conn.send(
            {
                "pose": (rvec, tvec),
                "corners": corners,
                "success": success,
                "capture_time": capture_time
            }
        )       
    
    pose_estimator.stop()


class PoseEstimatorExecutor:
    def __init__(self, clock, O_T_C, marker_length, use_marker_target=False, out_dir="out", dt = 0.01):
        self.out_dir = os.path.join(os.getcwd(), out_dir, "rhtoppra_camera_images")
        self.stop_event = mp.Event()

        self.use_marker_target = use_marker_target
        self.O_T_C = O_T_C

        self.conn1, self.conn2 = mp.Pipe(duplex=True)
        self.process = mp.Process(
            target=start_pose_estimation_loop, args=(self.stop_event, clock, self.conn2, self.out_dir, marker_length)
        )

        self.rvec = None
        self.tvec = None

        self.corners = None

        self.O_T_OBJ = None
        self.capture_time = None
        self.new_data = False
        self.time_stamps = []

        with open('captured_poses.pkl', 'rb') as f:
            captured_poses = pkl.load(f)
        
        self.captured_positions = np.array([T[:3,3] for _,_,T,_,_,_,_,_ in captured_poses])
        self.captured_times = np.array([t for t, _,_,_,_,_,_,_ in captured_poses])

        self.kalman_filter = [KalmanFilter (dim_x=3, dim_z=1),
                                KalmanFilter (dim_x=3, dim_z=1),
                                KalmanFilter (dim_x=3, dim_z=1)]
        r = [0.01, 0.01, 0.2] # measurement uncertainty
        for k in range(3):
            self.kalman_filter[k].F = np.array ([[1., dt, dt**2/2],
                                                [0., 1., dt],
                                                [0., 0., 1.]])
            self.kalman_filter[k].H = np.array ([[1., 0., 0.]]) # measurement function
            self.kalman_filter[k].P *= 1000. # covariance matrix
            self.kalman_filter[k].R = r[k] # measurement uncertainty
            self.kalman_filter[k].Q = Q_discrete_white_noise (dim=3, dt=dt, var=100) # process uncertainty


    def __enter__(self):

        self.process.start()
        self.get_pose(time=0, blocking=True)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
            
        self.stop_event.set()
        
        time.sleep(0.2)  # give some time to stop

        self.process.join()

    def get_pose(self, time, blocking=True):
        success = False

        corners = None
        if blocking:
            # wait for new data while blocking
            data = self.conn1.recv()
            rvec, tvec = data["pose"]
            corners = data["corners"]
            success = data["success"]
            self.capture_time = data["capture_time"]
            self.new_data = True
        else:
            # check if new data is available without blocking
            if self.conn1.poll():
                data = self.conn1.recv()
                rvec, tvec = data["pose"]
                corners = data["corners"]
                success = data["success"]
                self.capture_time = data["capture_time"]
                self.new_data = True
            else:
                self.new_data = False

        if not self.use_marker_target:
            # SIMULATION MODE - use prerecorded trajectory
            success = True
            self.new_data = True
            self.capture_time = time

            idx = np.searchsorted(self.captured_times, time + 3.5)
            if idx >= len(self.captured_times):
                idx = len(self.captured_times) - 1
            tvec = self.captured_positions[idx].reshape((3,1))
            t_vec_filt = tvec

            self.O_T_OBJ = np.array(
                [
                    [1.0, 0, 0, tvec[0][0]],
                    [0, -1.0, 0, tvec[1][0]],
                    [0, 0, -1.0, tvec[2][0]],
                    [0, 0, 0, 1.0],
                ]
            )

            if time < 0.3:
                self.O_T_OBJ = np.array(
                    [
                        [1.0, 0, 0, 0.5],
                        [0, -1.0, 0, -0.4],
                        [0, 0, -1.0, 0.2],
                        [0, 0, 0, 1.0],
                    ]
                )
            elif time < 0.6:
                self.O_T_OBJ = np.array(
                    [
                        [1.0, 0, 0, 0.5],
                        [0, -1.0, 0, 0.0],
                        [0, 0, -1.0, 0.2],
                        [0, 0, 0, 1.0],
                    ]
                )
            elif time < 0.9:
                self.O_T_OBJ = np.array(
                    [
                        [1.0, 0, 0, 0.5],
                        [0, -1.0, 0, 0.4],
                        [0, 0, -1.0, 0.2],
                        [0, 0, 0, 1.0],
                    ]
                )
            elif time < 1.2:
                self.O_T_OBJ = np.array(
                    [
                        [1.0, 0, 0, 0.5],
                        [0, -1.0, 0, 0.4],
                        [0, 0, -1.0, 0.5],
                        [0, 0, 0, 1.0],
                    ]
                )
            elif time < 1.5:
                self.O_T_OBJ = np.array(
                    [
                        [1.0, 0, 0, 0.5],
                        [0, -1.0, 0, 0.0],
                        [0, 0, -1.0, 0.5],
                        [0, 0, 0, 1.0],
                    ]
                )
            elif time < 1.8:
                self.O_T_OBJ = np.array(
                    [
                        [1.0, 0, 0, 0.5],
                        [0, -1.0, 0, -0.4],
                        [0, 0, -1.0, 0.5],
                        [0, 0, 0, 1.0],
                    ]
                )
            else:
                self.O_T_OBJ = np.array(
                    [
                        [1.0, 0, 0, 0.5],
                        [0, -1.0, 0, -0.4],
                        [0, 0, -1.0, 0.2],
                        [0, 0, 0, 1.0],
                    ]
                )

        else:
            # PERCEPTION MODE - use marker detection
            if success:
                if self.rvec is None or self.tvec is None:
                    # first data received
                    self.tvec = tvec
                    self.rvec = rvec
                    # initialize the kalman filter
                    for k in range(3):
                        self.kalman_filter[k].x = np.array ([self.tvec[k][0], 0, 0]) # initial state
                else:
                    if self.new_data:
                        self.tvec = tvec
                        self.rvec = rvec

                        # filter update
                        for k, filter in enumerate(self.kalman_filter):
                            filter.update(self.tvec[k][0])

            # filter predict
            t_vec_filt = np.zeros_like(self.tvec)
            for k, filter in enumerate(self.kalman_filter):
                filter.predict()
                t_vec_filt[k][0] = filter.x[0]


            self.O_T_OBJ = object_pose(np.zeros_like(self.rvec), t_vec_filt, self.O_T_C)
            self.O_T_OBJ[0:3,0:3] = np.array([[1.0, 0, 0],
                                             [0, -1.0, 0],
                                             [0, 0, -1.0]])
            self.O_T_OBJ[0:3, 3] += np.array([0.0, 0.0, 0.10])  # offset if needed

        # return the latest pose -- may be old if non-blocking and no new data is available
        return self.new_data, self.O_T_OBJ, self.tvec, t_vec_filt, corners, self.capture_time, success
