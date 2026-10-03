# -*- coding: utf-8 -*-
'''
Created on 28-June-2024 18:01
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''

from multiprocessing import Process, Manager
from src.src.core.machine_learning.computer_vision.store_tracker import StoreTracker
from src.database.schemas.process_logger_schema import ProcessLogger
from src.database.schemas.cameras_schema import Cameras
from collections import defaultdict, deque

import numpy as np
import math
import subprocess
import time
import os
import sys


class ProcessMonitor:
    def __init__(self, orchestrator, min_fps=3, window_size=21, fps_data_point_limit=15, error_limit=5, error_time_limit=20) -> None:
        """
        This class will store the processing info of all the process ran by the orchestrator.
        Mainly all the errors and current fps of processes. if error occurs then restart the process, if multiple error
        occurs in 5 second then restart whole service.

        It will also store fps for all the process for 20 (window size) data points. if the average fps of the system reduce
        below the fps_limit then it restarts the whole service

        It will also manage the store tracker kpis as its monitoring all the processes and its data

        Args:
            orchestrator (object): orchestrator self object to use its methods and variables
        """
        self.orchestrator = orchestrator
        self.shared_dict = self.orchestrator.shared_dict
        self.window_size = window_size
        self.error_limit = error_limit
        self.error_time_limit = error_time_limit
        self.fps_data_point_limit = fps_data_point_limit
        self.min_fps = min_fps
        self.process_data = defaultdict(lambda: {
            "error": deque(maxlen=error_limit),
            "error_time": deque(maxlen=error_limit),
            "fps": deque(maxlen=self.fps_data_point_limit)
        })
        self.port = 9990
        self.store_tracker = StoreTracker(self.orchestrator.device_names, window_size=30) # this window_size is different then class window_size

    def restart_process(self, device_name):

        # TODO: check it fps is not been stored
        # store shared dict 
        ProcessLogger.store(self.shared_dict[device_name])
        
        ## restart the process
        # Process dict
        self.process_dict = self.orchestrator.manager.dict()
        self.process_dict["run"] = False
        self.process_dict["error"] = []
        self.process_dict["fps"] = {}
        self.process_dict["data"] = {}
        self.process_dict["process_name"] = [device_name]

        self.shared_dict[device_name] = self.process_dict
        self.shared_dict[device_name]["run"] = True 
        # device started flag
        # start multiple processing camera streams and its processes 
        active_cameras = Cameras.objects(active=True, device_name=device_name).only('camera_address', 'device_name', 'zones', 'process_skip_frame').first()

        orchestrator_process = Process(target=self.orchestrator.start_process, args=(active_cameras, self.shared_dict))
        orchestrator_process.start()
        self.orchestrator.orchestrator_process_list[device_name] = orchestrator_process

    def restart_service(self):
        try:
                # Find the PIDs of the processes using the self.port
                lsof_cmd = f"lsof -t -i :{self.port}"
                result = subprocess.run(lsof_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"No process is using self.port {self.port}.")
                    return
                
                pids = result.stdout.strip().split()
                
                if pids:
                    # Kill each process using its PID
                    for pid in pids:
                        kill_cmd = f"kill -9 {pid}"
                        subprocess.run(kill_cmd, shell=True)
                        print(f"Process with PID {pid} on self.port {self.port} has been killed.")
                else:
                    print(f"No process is using self.port {self.port}.")

        except Exception as e:
            print(f"An error occurred: {e}")

    def monitor_processes(self):

        self.orchestrator.database.connect_db()

        while self.shared_dict["orchestrator"]["run"] == 1:
            self.store_tracker.update(self.shared_dict)

            # monitor all the processes and and find out if there is an error if yes then restart the process
            for device_name in self.orchestrator.device_names:
                # print("data: ", dict(self.shared_dict[device_name]))

                if device_name in self.shared_dict and "error" in self.shared_dict[device_name]:

                    error_list = self.shared_dict[device_name]["error"]
                    # print("error_list: ",error_list)

                    if len(error_list) > 0:

                        self.process_data[device_name]['error'].append(error_list[0])
                        self.process_data[device_name]['error_time'].append(time.time())


                        if len(self.process_data[device_name]['error']) == self.error_limit:
                            time_delta = self.process_data[device_name]['error_time'][0] - self.process_data[device_name]['error_time'][-1]

                            if time_delta < self.error_time_limit:
                                print("Restarting the service!")
                                print("process_data: ", self.process_data)
                                ProcessLogger.store(self.shared_dict[device_name])

                                self.restart_service()


                        print('Restarting the process for:', device_name)
                        self.restart_process(device_name)



                # check the fps reduction
                if device_name in self.shared_dict and "data" in self.shared_dict[device_name] and \
                    "fps" in self.shared_dict[device_name]["data"] and "process_fps" in self.shared_dict[device_name]["data"]["fps"]:
                    if self.shared_dict[device_name]["data"]["fps"]["process_fps"] != None:

                        # print("process_data1: ", self.process_data)

                        process_fps = int(float(self.shared_dict[device_name]["data"]["fps"]["process_fps"]))

                        self.process_data[device_name]['fps'].append(process_fps)

                        if len(self.process_data[device_name]['fps']) == self.fps_data_point_limit and np.array(self.process_data[device_name]['fps']).mean() < self.min_fps:
                            

                            #TODO: add reduction in apis 
                            ProcessLogger.store(self.shared_dict[device_name])

                            # print("process_data2: ", self.process_data)
                            # time.sleep(2)
                            # self.restart_service()

                # store process logs for every 1 mins
                        
            