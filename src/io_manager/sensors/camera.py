import threading
import time
import cv2
import subprocess
import numpy as np
import urllib

class IP_Camera:
    config = dict()
    def __init__(self, ip_address, device_name="camera-001", zones=[], process_skip_frame=1):
    
        self.ip_address = ip_address
        self.device_name = device_name
        self.zones = zones
        self.process_skip_frame = process_skip_frame

        self.is_open = False
        self.capture = None
        self.rgb_img = None
        self.fps = None
        self.cam_result = None
        self.plotted_frame = []
        self.frame_number = None
        # self.connect() 

    def connect(self):
        # Connect to the camera, initialize as needed
        # video_path="\one-by-one-person-detection.mp4"
        self.capture = cv2.VideoCapture(self.ip_address)  # Adjust the index based on your camera
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"Connected to camera: {self.ip_address} Fps: {self.fps}")

    def get_frame(self):

        if not self.is_open:
            self.is_open = True
            self.capture = cv2.VideoCapture(self.ip_address)  # Adjust the index based on your camera
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            print(f"Connected to camera: {self.ip_address} Fps: {self.fps}")
        else:
            print("camera already connected")
        start_time = time.time()

        while self.is_open:
            self.cam_result, self.rgb_img = self.capture.read()
            frame_number = self.capture.get(1)
            if frame_number >= 5:

                img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)
                _, frame_bytes = cv2.imencode('.jpg', img)
                print("image: ", img.shape)
                break
            else: 
                frame_bytes = []
                print("frame number is less than 5")
                if time.time() - start_time > 3:
                    break
        
        return frame_bytes


    def disconnect(self):
        # Disconnect or release any resources
        if self.capture:
            self.capture.release()

    def start_stream_read(self):
        while self.is_open:
            self.cam_result, self.rgb_img = self.capture.read()
            self.frame_number = self.capture.get(1)

            # print("flag", self.is_open, self.cam_result)
            
            if not self.cam_result:
                print("error: ", self.cam_result, self.rgb_img)
                break
                continue

            if "rtsp" not in self.ip_address:
                if self.fps:
                    # print("video and skiping")
                    time.sleep(1/self.fps)
                    
        print("Streaming stopped!", self.is_open, self.cam_result)
        # subprocess.run("lsof -t -i :9990 | xargs kill -9", shell=True)

    def start_streaming(self):

        thread = threading.Thread(target=self.start_stream_read)
        thread.start()

        # return self.stream_to_frontend()

    def stream_to_frontend(self):

        while self.is_open:
            
            if self.rgb_img is None:
                self.rgb_img = cv2.imread("src/res/desktop.jpg")
                self.rgb_img = cv2.resize(self.rgb_img, (1280, 720))
            
            if len(self.plotted_frame) == 0:
                self.plotted_frame = self.rgb_img

            # print(f"self.plotted_frame.shape : {self.plotted_frame.shape}")
            _, frame_bytes = cv2.imencode('.jpg', self.plotted_frame)
            self.frame_bytes = frame_bytes.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + self.frame_bytes + b'\r\n\r\n')

   
    def stop_streaming(self):
        self.is_open = False
        if self.capture:
            self.capture.release()
            
            
# if __name__ == "__main__":
#     ip_cam = IP_Camera({})
#     ip_cam.start_streaming()
    