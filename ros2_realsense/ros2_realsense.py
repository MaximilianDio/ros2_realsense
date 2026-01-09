import rclpy
import cv2
from rclpy.node import Node
from ros2_realsense.RealSenseCapture import RealsenseCapture

#!/usr/bin/env python3
"""
ROS2 node to capture images from Intel RealSense camera using RealsenseCapture class.
Saves images to disk and optionally displays them in a window.

Usage:
    ros2 run ros2_realsense ros2_realsense --ros-args -p show_frame:=True -p save_frame:=True -p out_dir:=./out/test
    
    Alternatively, parameters set parameter during run time.
"""


class RealsenseNode(Node):
    def __init__(self):
        super().__init__('realsense_node')
        
        # Declare parameters with default values
        self.declare_parameter('show_frame', False)
        self.declare_parameter('save_frame', False)
        self.declare_parameter('out_dir', './image_capture/test')
        
        # Get parameter values
        show_frame = self.get_parameter('show_frame').value
        save_frame = self.get_parameter('save_frame').value
        out_dir = self.get_parameter('out_dir').value
        
        # Create RealsenseCapture object
        self.realsense = RealsenseCapture(out_dir=out_dir)
        
        # Create timer for periodic capture (e.g., 60 Hz)
        self.timer_period = 1.0 / 60.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.t0 = None
        
        self.capture_idx = 0
        
        self.get_logger().info(f'Realsense node started with show_frame={show_frame}, save_frame={save_frame}, out_dir={out_dir}')
    
    def timer_callback(self):
        

        # Call capture with time and index
        try:
            image, capture_time = self.realsense.capture()
            
            timestamp = 0        

            # Save image (raw) if requested 
            if self.get_parameter('save_frame').value:
                if self.t0 is None:
                    self.get_logger().info('Start saving frames to disk.')
                    self.t0 = self.get_clock().now().nanoseconds / 1e9  # seconds
        
                # Get ROS2 time
                current_time = self.get_clock().now().nanoseconds / 1e9 # Convert to seconds
                timestamp = current_time - self.t0  # relative time since start

                self.realsense.save_image(timestamp, self.capture_idx, image)
                
            # Display frame if requested (minimal overhead)
            if self.get_parameter('show_frame').value:
                
                # add time stamp to image
                formatted_timestamp = f"{int(timestamp*1e3):04d}"
                cv2.putText(
                    image,
                    f"Time: {formatted_timestamp} ms",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                
                cv2.imshow("RealSense", image)
                cv2.waitKey(1)

            self.capture_idx += 1
        except Exception as e:
            self.get_logger().error(f'Capture failed: {str(e)}')
    
    def destroy_node(self):
        # Clean up
        if hasattr(self.realsense, 'stop'):
            self.realsense.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = RealsenseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()