import cv2
import os
import numpy as np
from natsort import natsorted
from RealsensePoseEstimator import ArucoPose

"""
Generate a video from a sequence of images stored in a folder.

Usage:
    python generate_video.py <image_folder> <output_video> [fps]
    - image_folder: Path to folder containing .png images
    - output_video: Path for output video file
    - fps: (optional) Frames per second (default: 30)
"""

def object_pose(rvec, tvec, M_T_C):
    T_M = np.eye(4) # Transformation matrix of marker in world frame
    R = cv2.Rodrigues(rvec)[0]
    T_M[0:3, 0:3] = R
    T_M[0:3, 3] = tvec.flatten()
    
    T_C = T_M @ M_T_C
    rvec_C, _ = cv2.Rodrigues(T_C[0:3, 0:3])
    tvec_C = T_C[0:3, 3].reshape((3, 1))
    return rvec_C, tvec_C


def generate_video_from_images(image_folder, output_video, fps=30):
    # Get all .png files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natsorted(images)  # Sort images naturally

    if not images:
        print("No .png files found in the folder.")
        return

    images_timestamps = [
        float(image.split("_")[-1].replace(".png", "")) * 1e3  # Extract timestamp
        for image in images
    ]

    # there are multiple images with 0 and end timestamp, so we remove duplicates
    idx_first = max(0,next(i for i, v in enumerate(images_timestamps) if v != 0) - 1)
    idx_last = next(
        i for i, v in enumerate(images_timestamps) if v == images_timestamps[-1]
    )

    images_timestamps = images_timestamps[idx_first : idx_last + 1]
    images = images[idx_first : idx_last + 1]
    np.diff(images_timestamps)  # just to check if timestamps are increasing
    fps_mean = 1000.0 / np.mean(np.diff(images_timestamps))  # in ms

    images_timestamped = list(zip(images, images_timestamps))

    t_end = images_timestamps[-1]
    fps = len(images_timestamped) / (t_end / 1000.0)

    print(
        f"Generating video at {fps:.2f} FPS, FPS mean {fps_mean:.2f} from {len(images_timestamped)} images..."
    )

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    prev_frame = None
    prev_time = None  # in milliseconds

    intrinsics = np.loadtxt("out/test_camera_images/camera_matrix.txt")
    dist = np.loadtxt("out/test_camera_images/dist_coeffs.txt")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    aruco_pose = ArucoPose(intrinsics, dist, detector, marker_length=0.04)

    # transformation from marker to object (example values)
    M_T_C = np.array([[1, 0, 0, 0.18],
                    [0, 1, 0, 0.00], 
                    [0, 0, 1, -0.05],
                    [0, 0, 0, 1]])
    
    t_vec_C_list = []

    for image, t_ist in images_timestamped:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)


        corners, ids, success, (rvec_M, tvec_M) = aruco_pose.get_pose(frame)

        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvec_C, tvec_C = object_pose(rvec_M, tvec_M, M_T_C)
        cv2.drawFrameAxes(frame, intrinsics, dist, rvec_C, tvec_C, 0.03)

        t_vec_C_list.append(tvec_M)

        # 3d projection of t_vec_C
        for i in range(1, len(t_vec_C_list)):
            p1, _ = cv2.projectPoints(t_vec_C_list[i-1], np.zeros((3,1)), np.zeros((3,1)), intrinsics, dist)
            p2, _ = cv2.projectPoints(t_vec_C_list[i], np.zeros((3,1)), np.zeros((3,1)), intrinsics, dist)
            cv2.line(frame, (int(p1[0][0][0]), int(p1[0][0][1])), (int(p2[0][0][0]), int(p2[0][0][1])), (0, 255, 0), 2)

        # overlay timestamp
        formatted_timestamp = f"{int(t_ist):04d}"
        cv2.putText(
            frame,
            f"Time: {formatted_timestamp} ms",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # first frame → just write it once
        if prev_time is None:
            video.write(frame)
            prev_frame = frame
            prev_time = t_ist
            continue

        # -------------------------------
        # Compute time difference
        # -------------------------------
        dt = (t_ist - prev_time) / 1000.0  # convert ms → seconds
        repeat_count = max(1, int(round(dt * fps_mean)))

        # print(f"t_ist: {t_ist:.2f} ms, dt: {dt:.3f} s, repeat_count: {repeat_count}")
        # -------------------------------
        # Write the PREVIOUS frame repeat_count times
        # This fills gaps if timestamps jump
        # -------------------------------
        for _ in range(repeat_count):
            video.write(prev_frame)

        # Now update the previous frame & time
        prev_frame = frame
        prev_time = t_ist

        cv2.imshow("Annotated Image", frame)
        cv2.waitKey(1)

    # Write the final frame once
    video.write(prev_frame)

    video.release()
    print(f"Video saved as {output_video}")


def main(args=None):
    
    image_folder = "out/test_camera_images"
    output_video = "out/video_output.mp4"
    fps = 60

    print(f"Generating video from images in {image_folder}...")
    generate_video_from_images(image_folder, output_video, fps)


if __name__ == "__main__":
    main()
