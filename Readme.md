# Installation

## dependencies

install required packages:
```bash
python3 -m pip install -r ./src/ros2_realsense/requirements.txt
```

## ros build

build the workspace (assuming you are in the root of the workspace):
```bash
colcon build --symlink-install --packages-select ros2_realsense
```

# Usage

## calibration

run the ros2 node to perform camera calibration:
```bash
ros2 run ros2_realsense aruco_calibration \
  --ros-args -p config_file:=/workspaces/src/ros2_realsense/config/aruco_calibration_config.yaml
```

## create images

run the ros2 node to capture images from the RealSense camera:
```bash
ros2 run ros2_realsense ros2_realsense --ros-args -p show_frame:=True -p save_frame:=True -p out_dir:=./out/test
```
- parameters:
    - `show_frame` (bool): whether to display the captured frames in a window.
    - `save_frame` (bool): whether to save the captured frames to disk.
    - `out_dir` (str): directory to save the captured frames.

## generate video from images

run the following command to create a video from the saved images:
```bash
ros2 run ros2_realsense generate_video <input_image_dir> <output_video_file>
```
- parameters:
    - `<input_image_dir>`: directory containing the saved images.
    - `<output_video_file>`: path to save the generated video file.