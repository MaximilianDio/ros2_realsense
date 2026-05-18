import os
import cv2
import time
import numpy as np
import multiprocessing as mp
import argparse
import pyrealsense2 as rs

class RealsenseCapture:
    """Handle RealSense camera capture and configuration."""
    
    def __init__(self, out_dir, width=640, height=480, fps=60):
        """
        Initialize RealSense camera with optimized settings.
        
        Args:
            out_dir: Output directory for saving images and camera parameters
        """

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
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)  
        try:
            cfg = self.pipe.start(config)
        except RuntimeError as e:
            print(f"Error configuring RealSense stream: {e}")
            raise

        
         
        # Optimize camera settings for low latency
        sensor = cfg.get_device().first_color_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto-exposure for consistent timing
        sensor.set_option(rs.option.white_balance, 3000)
        sensor.set_option(rs.option.exposure, 25)  # Fast exposure (1-20ms range)
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
            image, capture_time = camera.capture()

            # print current clock time and capture time
            print(f"Clock: {clock.value:.3f} s, Capture Time: {capture_time:.3f} ms", end='\r')
            
            if image is not None:

                # Save to disk if requested
                if save_frame:
                    camera.save_image(clock.value, idx, image)
                idx += 1

                                # Display frame if requested
                if show_frame:
                    # add time stamp to image
                    formatted_timestamp = f"{int(clock.value*1e3):04d}"
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
    
    parser = argparse.ArgumentParser(description="RealSense camera capture with optional display and saving")
    parser.add_argument("--out-dir", type=str, default="test_camera_images", help="Output directory for images")
    parser.add_argument("--show-frame", action="store_true", default=False, help="Display captured frames")
    parser.add_argument("--save-frame", action="store_true", default=False, help="Save frames to disk")
    
    args = parser.parse_args()
    
    # Shared clock for synchronization across processes
    clock = mp.Value('d', 0.0)
    
    with RealSenseCaptureExecutor(
        clock=clock, 
        out_dir="out/" + args.out_dir, 
        show_frame=args.show_frame, 
        save_frame=args.save_frame
    ) as executor:
        t0 = time.time()
        try:
            while True:
                clock.value = time.time() - t0
                time.sleep(0.001)  # 1ms sleep for ~1kHz clock update
        except KeyboardInterrupt:
            print("\nShutting down...")