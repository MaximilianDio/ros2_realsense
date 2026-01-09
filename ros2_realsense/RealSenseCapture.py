import os
import cv2
import time
import numpy as np
import psutil
import multiprocessing as mp

def print_memory_usage():
    """Print current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

class RealsenseCapture:
    """Handle RealSense camera capture and configuration."""
    
    def __init__(self, out_dir):
        """
        Initialize RealSense camera with optimized settings.
        
        Args:
            out_dir: Output directory for saving images and camera parameters
        """
        import pyrealsense2 as rs

        # Setup output directory
        self.out_dir = os.path.join(os.getcwd(), out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Clear existing files (more efficient with pathlib or direct iteration)
        for file in os.scandir(self.out_dir):
            if file.is_file():
                os.unlink(file.path)

        # Configure and start pipeline
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  
        # config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)  
        # config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30 ) 
        cfg = self.pipe.start(config)
         
        # Optimize camera settings for low latency
        sensor = cfg.get_device().first_color_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto-exposure for consistent timing
        sensor.set_option(rs.option.white_balance, 3000)
        sensor.set_option(rs.option.exposure, 20)  # Fast exposure (1-20ms range)
        sensor.set_option(rs.option.gain, 128)  # Compensate for low exposure
        
        # Extract and save camera intrinsics
        self.intr = cfg.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.mtx = np.array([[self.intr.fx, 0, self.intr.ppx],
                             [0, self.intr.fy, self.intr.ppy],
                             [0, 0, 1]])
        self.dist = np.array(self.intr.coeffs)

        # Save calibration parameters once during initialization
        np.savetxt(os.path.join(self.out_dir, "camera_matrix.txt"), self.mtx)
        np.savetxt(os.path.join(self.out_dir, "dist_coeffs.txt"), self.dist)

    def capture(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            tuple: (image array, timestamp in ms)
        """
        # Blocking call - waits for next frame
        frames = self.pipe.wait_for_frames()
        capture_time = frames.get_timestamp()
        frame = frames.get_color_frame()

        if frame:
            # Zero-copy conversion to numpy array
            image = np.asanyarray(frame.get_data())

            return image, capture_time
        else:
            print("Frame is invalid or dropped.")
            return None, None
    
    def save_image(self, time, idx, image):
        """
        Save image to disk with timestamp in filename.
        
        Args:
            time: Timestamp for filename
            idx: Frame index
            image: Image array to save
        """
        filename = f"image_{idx:04d}_{time:.3f}.png"
        image_filename = os.path.join(self.out_dir, filename)
        cv2.imwrite(image_filename, image)

    def stop(self):
        """Stop the RealSense pipeline and release resources."""
        self.pipe.stop()
        cv2.destroyAllWindows()  # Clean up any open windows

def start_capture_loop(stop_event, clock, out_dir, show_frame, save_frame):
    """
    Main capture loop running in separate process.
    
    Args:
        stop_event: Multiprocessing event to signal shutdown
        clock: Shared memory value for synchronized timing
        out_dir: Directory for saving images
        show_frame: Whether to display frames
        save_frame: Whether to save frames to disk
    """
    camera = RealsenseCapture(out_dir=out_dir)
    idx = 0
    
    while not stop_event.is_set():
        try:
            # Capture frame
            image, capture_time = camera.capture(show_frame)
            
            if image is not None:
                # Save to disk if requested
                if save_frame:
                    camera.save_image(capture_time, idx, image)
                idx += 1
                
        except Exception as e:
            print(f"Error during capture: {e}")
            break

    print(f"Capture loop ended: {idx} frames captured")
    camera.stop()

class RealSenseCaptureExecutor:
    """Context manager for running camera capture in separate process."""
    
    def __init__(self, clock, out_dir, show_frame=False, save_frame=True):
        """
        Initialize capture executor.
        
        Args:
            clock: Shared memory value for timing synchronization
            out_dir: Output directory for images
            show_frame: Display captured frames
            save_frame: Save frames to disk
        """
        self.stop_event = mp.Event()
        self.process = mp.Process(
            target=start_capture_loop, 
            args=(self.stop_event, clock, out_dir, show_frame, save_frame)
        )

    def __enter__(self):
        """Start capture process."""
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop capture process and wait for cleanup."""
        self.stop_event.set()
        self.process.join(timeout=5.0)  # Add timeout to prevent hanging
        if self.process.is_alive():
            self.process.terminate()

if __name__ == "__main__":
    # Shared clock for synchronization across processes
    clock = mp.Value('d', 0.0)
    
    with RealSenseCaptureExecutor(
        clock=clock, 
        out_dir="out/test_camera_images", 
        show_frame=True, 
        save_frame=True
    ) as executor:
        t0 = time.time()
        try:
            while True:
                clock.value = time.time() - t0
                time.sleep(0.001)  # 1ms sleep for ~1kHz clock update
        except KeyboardInterrupt:
            print("\nShutting down...")