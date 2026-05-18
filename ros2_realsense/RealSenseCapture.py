import os
import cv2
import time
import numpy as np
import multiprocessing as mp
import argparse
from concurrent.futures import ThreadPoolExecutor
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
        filename = f"image_{idx:04d}_{time:.3f}.jpg"
        image_filename = os.path.join(self.out_dir, filename)
        cv2.imwrite(image_filename, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def stop(self):
        """Stop the RealSense pipeline and release resources."""
        self.pipe.stop()
        cv2.destroyAllWindows()  # Clean up any open windows

class AsyncImageWriter:
    """
    Decouples disk writes from the capture thread using a thread pool.

    The capture thread calls submit() which returns in < 0.1 ms; worker
    threads handle JPEG compression and disk I/O in the background.
    """

    def __init__(self, out_dir, num_workers=2):
        self._out_dir = out_dir
        self._pool = ThreadPoolExecutor(max_workers=num_workers)
        self._futures = []

    def submit(self, idx, timestamp, image):
        """Enqueue a frame for writing. Returns immediately."""
        future = self._pool.submit(self._write, idx, timestamp, image.copy())
        self._futures.append(future)

    def _write(self, idx, timestamp, image):
        path = os.path.join(self._out_dir, f"image_{idx:04d}_{timestamp:.3f}.jpg")
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def pending(self):
        """Return number of writes not yet completed."""
        return sum(1 for f in self._futures if not f.done())

    def flush(self):
        """Block until all queued writes are complete."""
        for f in self._futures:
            f.result()
        self._pool.shutdown(wait=False)


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
    writer = AsyncImageWriter(out_dir=camera.out_dir) if save_frame else None
    idx = 0
    clock0 = clock.value
    while not stop_event.is_set():

        try:
            # Capture frame
            image, capture_time = camera.capture()

            # print current clock time and capture time

            print(f"Clock: {clock.value:.3f} s, Capture Time: {capture_time:.3f} ms, fps: {idx/(clock.value-clock0)}", end='\r')
            
            if image is not None:

                # Submit to async writer — returns in < 0.1 ms
                if save_frame:
                    writer.submit(idx, clock.value, image)

                    # Warn if writer is falling behind
                    if idx % 30 == 0:
                        pending = writer.pending()
                        if pending > 60:
                            print(f"\n[WARN] Writer backlog: {pending} frames pending")

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

    if save_frame:
        print(f"\nFlushing {writer.pending()} pending writes...")
        writer.flush()
    print(f"Capture loop ended: {idx} frames captured")
    camera.stop()

class RealSenseCaptureExecutor:
    """Context manager for running camera capture in separate process."""
    
    def __init__(self, clock, out_dir, show_frame=False, save_frame=True,
                 width = 1920, height = 1080, fps = 30):
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
            args=(self.stop_event, clock, out_dir, show_frame, save_frame, width, height, fps)
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
        save_frame=args.save_frame,
        width=960,
        height=540,
        fps=30
    ) as executor:
        t0 = time.time()
        try:
            while True:
                clock.value = time.time() - t0
                time.sleep(0.001)  # 1ms sleep for ~1kHz clock update
        except KeyboardInterrupt:
            print("\nShutting down...")