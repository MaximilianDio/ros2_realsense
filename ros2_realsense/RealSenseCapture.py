import os
import cv2
import time
import numpy as np
import psutil

import multiprocessing as mp

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

class RealsenseCapture:
    def __init__(self, out_dir):
        import pyrealsense2 as rs

        self.out_dir = os.path.join(os.getcwd(), out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        # clear existing files in the directory
        for file in os.scandir(self.out_dir):
            # Delete only files, not directories
            if file.is_file():
                # Delete the file
                os.unlink(file.path)

        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  
        # config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)  
        # config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30 ) 
        cfg = self.pipe.start(config)
         
        # Set camera settings for faster exposure and gain
        sensor = cfg.get_device().first_color_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.white_balance, 3000)
        sensor.set_option(rs.option.exposure, 30)  # Lower value for faster exposure (try 1-20)
        sensor.set_option(rs.option.gain, 64)  # Increase gain if image is too dark (try 16-64)
        
       

        self.intr = cfg.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.mtx = np.array([[self.intr.fx, 0, self.intr.ppx],
                        [0, self.intr.fy, self.intr.ppy],
                        [0, 0, 1]])
        self.dist = np.array(self.intr.coeffs)

        # store coefficients in out_dir for later use
        np.savetxt(os.path.join(self.out_dir, "camera_matrix.txt"), self.mtx)
        np.savetxt(os.path.join(self.out_dir, "dist_coeffs.txt"), self.dist)
        


        self.idx = 0
        self.image = None
        self.capture_time = None
        self.first_capture_time = None

    def capture(self, show_frame):
        # this is a blocking call - waits for the next set of frames
        frames = self.pipe.wait_for_frames()

        capture_time = frames.get_timestamp()
        frame = frames.get_color_frame()

        if frame:
            self.image = np.asanyarray(frame.get_data())
            self.capture_time = capture_time

            # record the first capture time
            if self.first_capture_time is None:
                self.first_capture_time = self.capture_time

            # show the captured frame
            if show_frame:
                cv2.imshow("RealSense", self.image)
                cv2.waitKey(1)

        else:
            print("Frame is invalid or dropped.")

        return self.image, self.capture_time
    
    def save_image(self, time, idx, image):
        filename = f"image_{idx:04d}_{time:.3f}.png"
        image_filename = os.path.join(self.out_dir, filename)
        cv2.imwrite(image_filename, image)

    def stop(self):
        self.pipe.stop()

def start_capture_loop(stop_event, clock, out_dir, show_frame, save_frame):
    camera = RealsenseCapture(out_dir=out_dir)
    idx = 0
    while True:
        try:
            # print_memory_usage()
            image, capture_time = camera.capture(clock.value, idx, show_frame, save_frame)
            
            idx += 1
            # save image to disk
            if stop_event.is_set():
                print("Stopping capture loop...")
                break
        except Exception as e:
            print(f"Error during capture: {e}")
            break

    print("Capture loop ended: number of frames captured:", idx)
    camera.stop()

class RealSenseCaptureExecutor:
    def __init__(self, clock, out_dir, show_frame=False, save_frame=True):
       
        self.stop_event = mp.Event()
        self.process = mp.Process(
                target=start_capture_loop, args=(self.stop_event, clock, out_dir, show_frame, save_frame)
            )

    def __enter__(self):
        self.process.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.process.join()

if __name__ == "__main__":
    clock = mp.Value('d', 0.0)
    with RealSenseCaptureExecutor(clock=clock, out_dir="out/test_camera_images", show_frame=True, save_frame=True) as executor:
        t0 = time.time()
        while True:
            ti = time.time()
            clock.value = ti - t0
            time.sleep(0.001)