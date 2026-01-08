import rclpy
from rclpy.node import Node
from ros2_realsense.RealSenseCapture import RealsenseCapture
# from ros2_realsense.RealsensePoseEstimator import RealsensePoseEstimator

#!/usr/bin/env python3



class RealsenseNode(Node):
    def __init__(self):
        super().__init__('realsense_node')
        
        # Create RealsenseCapture object
        self.realsense = RealsenseCapture(out_dir="./out/test")  # specify output directory
        
        # Create timer for periodic capture (e.g., 60 Hz)
        self.timer_period = 1.0 / 60.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.t0 = None
        
        self.capture_idx = 0
        
        self.get_logger().info('Realsense node started')
    
    def timer_callback(self):
        
        if self.t0 is None:
            self.t0 = self.get_clock().now().nanoseconds / 1e9  # seconds
        
        # Get ROS2 time
        current_time = self.get_clock().now().nanoseconds / 1e9 # Convert to seconds
        timestamp = current_time - self.t0  # relative time since start
        
        # Call capture with time and index
        try:
            image, capture_time = self.realsense.capture(None, self.capture_idx, show_frame=True, save_frame=True)
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