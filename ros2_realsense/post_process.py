from RealsensePoseEstimator import ArucoPose
import numpy as np
import cv2

def object_pose(rvec, tvec, M_T_C):
    T_M = np.eye(4) # Transformation matrix of marker in world frame
    R = cv2.Rodrigues(rvec)[0]
    T_M[0:3, 0:3] = R
    T_M[0:3, 3] = tvec.flatten()
    
    T_C = T_M @ M_T_C
    rvec_C, _ = cv2.Rodrigues(T_C[0:3, 0:3])
    tvec_C = T_C[0:3, 3].reshape((3, 1))
    return rvec_C, tvec_C

# Load camera intrinsics and distortion coefficients
intrinsics = np.loadtxt("out/test_camera_images/camera_matrix.txt")
dist = np.loadtxt("out/test_camera_images/dist_coeffs.txt")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

aruco_pose = ArucoPose(intrinsics, dist, detector, marker_length=0.04)

image = cv2.imread("out/test_camera_images/image_0561_19.007.png")
corners, ids, success, (rvec_M, tvec_M) = aruco_pose.get_pose(image)

annotated_image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

cv2.drawFrameAxes(annotated_image, intrinsics, dist, rvec_M, tvec_M, 0.03)

# transformation from marker to object (example values)
M_T_C = np.array([[1, 0, 0, 0.18],
                  [0, 1, 0, 0.00], 
                  [0, 0, 1, -0.05],
                  [0, 0, 0, 1]])
rvec_C, tvec_C = object_pose(rvec_M, tvec_M, M_T_C)
cv2.drawFrameAxes(annotated_image, intrinsics, dist, rvec_C, tvec_C, 0.03)

cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()