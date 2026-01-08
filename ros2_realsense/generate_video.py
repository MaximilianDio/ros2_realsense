import cv2
import os
import numpy as np
from natsort import natsorted

"""
Generate a video from a sequence of images stored in a folder.
"""


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
    idx_first = next(i for i, v in enumerate(images_timestamps) if v != 0) - 1
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

    for image, t_ist in images_timestamped:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

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

    # Write the final frame once
    video.write(prev_frame)

    video.release()
    print(f"Video saved as {output_video}")


if __name__ == "__main__":
    image_folder = f"out/test"  # Replace with your folder path
    output_video = f"out/test/video.mp4"  # Replace with your desired output video name
    fps = 60  # Frames per second

    print("Generating video from captured images...")
    generate_video_from_images(image_folder, output_video, fps)
