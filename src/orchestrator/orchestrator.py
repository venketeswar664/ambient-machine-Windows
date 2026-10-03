from datetime import datetime
import time
from multiprocessing import Process, Manager
import cv2
import numpy as np
import os
import imageio
import sys
import copy
from src.comms.socket.socket_connection import SocketConnection
from src.src.core.machine_learning.machine_learning import MachineLearning



# from src.io_manager.io_manager import IOManage

from src.io_manager.sensors.cpu_usage import ResourceMonitor
from src.database.database_connector import Database
from src.database.schemas.cameras_schema import Cameras
from src.database.schemas.zone_schema import Zone
from src.database.schemas.store_schema import Store



from src.io_manager.sensors.camera import IP_Camera
from src.orchestrator.process_monitor import ProcessMonitor



class Orchestrator:

    _io_manager = None
    _state_manager = None
    _config = None
    _task_dag = None
     

    def __init__(self) -> None:
        """
        Initialization of all the variables for the orchestrator  
        """
        self.database = Database()
        self.database.connect_db()
        
        # multiprocessing shared dict
        self.manager = Manager()
        self.shared_dict = self.manager.dict()

        self.process_dict = self.manager.dict()
        self.process_dict["run"] = False
        self.process_dict["error"] = []
        self.process_dict["fps"] = {}
        self.process_dict["data"] = {}
        self.process_dict["process_name"] = ["orchestrator"]


        self.shared_dict["orchestrator"] = self.process_dict
        # run_analysis is the flag which is checked while processes
        
        self.frame_skip_factor = 1
        # resource_monitoring
        self.resource_monitor = ResourceMonitor()

        # video results dirs
        self.save_videos_flag = True
        self.results_base_dir = "./results/"

        # orchestrator run time
        self.run_time = datetime.now()
        self.run_time = self.run_time.strftime("%Y-%m-%d %H:%M:%S")

        # save video
        self.frame_size = (640, 360)
        self.frame_size = (1280, 720)

        self.fps = 30

        print("self.frame_size",  self.frame_size)
        # Resize background image to ensure consistent frame size
        self.background = cv2.resize(cv2.imread("src/res/desktop.jpg"), self.frame_size)

    def _set_video_writer(self):

        self.num_cameras = len(self.device_names)
        self.rows = 2  # Example: 3 self.rows
        self.cols = (self.num_cameras + self.rows - 1) // self.rows  # Calculate required columns

        # Initialize a black canvas for combined frame
        self.combined_height = self.frame_size[1] * self.rows
        self.combined_width = self.frame_size[0] * self.cols

        if self.save_videos_flag:
            output_path = self.results_base_dir  + f"{self.run_time}_processed.mp4"
            os.makedirs(self.results_base_dir, exist_ok=True)

            self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.combined_width, self.combined_height))


    def start(self):

        """This method will start orchestrator:
                1. Monitor system cpu, gpu, ram usage
                2. start orchestrator process (start camera and ml model processing and other kpi management)
                3. start monitoring orchestrator processes (1. get the data from the kpi produces by the process and combine it
                                                            2. if any process stops store the error and restart the process)
        """


        print("Starting Analysis on a different processes.") 

        self.device_names = list()
        self.orchestrator_process_list = dict()

        self.shared_dict["orchestrator"]["run"] = True # orchestrator started flag

        active_cameras = Cameras.objects(active=True).only('cameraAddress', 'device_name', 'zones', 'process_skip_frame', 'pipeline')
        
        try:
            for camera_doc in active_cameras:

                device_name = camera_doc.device_name

                self.device_names.append(device_name)

                # Process dict
                self.process_dict = self.manager.dict()
                self.process_dict["run"] = False
                self.process_dict["error"] = []
                self.process_dict["fps"] = {}
                self.process_dict["data"] = {}
                self.process_dict["process_name"] = [device_name]

                self.shared_dict[device_name] = self.process_dict
                self.shared_dict[device_name]["run"] = True 
                # device started flag
                # start multiple processing camera streams and its processes 

                print('shared_dict:', self.shared_dict)
                orchestrator_process = Process(target=self.start_process, args=(camera_doc, self.shared_dict))
                orchestrator_process.start()
                self.orchestrator_process_list[device_name] = orchestrator_process

        except Exception as e:
            print(f"An error occurred while processing cameras: {e}")

        self._set_video_writer()
                
                
        # Starting monitoring processes for GPU, CPU and RAM
        monitor_system = Process(target=self.resource_monitor.run_monitor)
        monitor_system.start()
        self.orchestrator_process_list["monitor_system"] = monitor_system

        # Process logger 
        # self.process_monitor = ProcessMonitor(self, window_size=21)
        # monitor_process = Process(target=self.process_monitor.monitor_processes)
        # monitor_process.start()
        # self.orchestrator_process_list["monitor_process"] = monitor_process

      

    def stop(self):
        print("stopping orchestrator")
        self.shared_dict["orchestrator"]["run"] = 0 # orchestrator STOP flag
        for device_name, orchestrator_process in self.orchestrator_process_list.items():
            orchestrator_process.join()

        self.out.release()

    # def start_camera_streaming(self, camera_doc, shared_dict):
    def start_process(self, camera_doc, shared_dict):
        """
        Variables to be used by human tracking

        """
        self.database.connect_db()

        camera_address, device_name, zones, process_skip_frame, pipeline = camera_doc.camera_address, \
                                                                    camera_doc.device_name, \
                                                                    camera_doc.zones, \
                                                                    camera_doc.process_skip_frame, \
                                                                    camera_doc.pipeline
                                                                            
        
        # load camera module and start streaming
        camera = IP_Camera(camera_address, device_name, zones, process_skip_frame)
        camera.connect()
        camera.is_open = True

        # create ML object and load models
        machine_learning = MachineLearning(camera, zones, pipeline)        
        camera.start_streaming()

        iteration_count = 0
        # try:
        if 1:
            while shared_dict["orchestrator"]["run"] and shared_dict[device_name]["run"]:            
                # print("camera results: ", camera.cam_result, camera.frame_number)
                if not camera.cam_result and not camera.frame_number:
                    print("Frame is none")
                    continue
                # print("camera.frame_number", camera.frame_number)
                # print("process_skip_frame", process_skip_frame)

                if camera.frame_number % process_skip_frame != 0:
                    # time.sleep(1/self.cam_fps)
                    continue  

                machine_learning.process(shared_dict)

                # test
                # if device_name == "camera-9" and camera.frame_number > 100:
                #     raise Exception("Manual error for testing")
                
                # if device_name == "camera-2" and iteration_count > 10:
                #     raise Exception("Manual error for testing")
                
                # if device_name == "camera-2" and iteration_count > 20:
                #     time.sleep(1)

                iteration_count += 1

                # break
              
        # except Exception as e:
            
        #     shared_dict[device_name]["error"] = [str(e)]
        #     self.out.release()
        #     print(f"Error while process : {e}")


        camera.stop_streaming()
        SocketConnection.send_status_update("cam-release")


    def stream_orchestrator(self):
        "monitor all the processes state and errors and record them if any process stops then restart it"

        # try:
        if 1:
            while self.shared_dict["orchestrator"]["run"] == 1:
                
                frame_bytes = self.stream_to_frontend()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        
        # except Exception as e:
        #     print(f"Exception while streaming : {e}") 
        #     self.out.release()

        # self.out.release()
        
    def stream_to_frontend(self):
    
        frames = []
        frame_bytes = None
        # while self.shared_dict["orchestrator"]["run"] == 1:
        for device_name in self.device_names:
            if device_name in self.shared_dict and "data" in self.shared_dict[device_name] and "frame" in self.shared_dict[device_name]["data"]:
                frame = self.shared_dict[device_name]["data"]["frame"]
                frame = cv2.resize(frame, self.frame_size)  # Resize to match the background/frame size

            else:

                frame = self.background

            frames.append(frame)

        if len(self.device_names) == 1:
            combined_frame =  cv2.resize(frame, (self.combined_height, self.combined_width))
        
        else:
            combined_frame = np.zeros((self.combined_height, self.combined_width, 3), dtype=np.uint8)

            # Place each frame in the correct position within the combined frame
            for idx, frame in enumerate(frames):
                row = idx // self.cols
                col = idx % self.cols
                y_start = row * self.frame_size[1]
                y_end = y_start + self.frame_size[1]
                x_start = col * self.frame_size[0]
                x_end = x_start + self.frame_size[0]
                combined_frame[y_start:y_end, x_start:x_end] = frame

        
        if self.save_videos_flag:
            # Save anonymized frame
            self.out.write(combined_frame)
        
        _, frame_bytes = cv2.imencode('.jpg', combined_frame)
        frame_bytes = frame_bytes.tobytes()

        return frame_bytes
    
