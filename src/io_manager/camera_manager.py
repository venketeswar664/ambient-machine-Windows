# -*- coding: utf-8 -*-
'''
Created on 11-May-2024 16:19
Project: Ambient-Machine
@author: Pranjal Bhaskare
@email: pranjal@neophyte.ai
'''

"""
A class to manage all the cameras to load and process
"""
####################### remove once complete
# import sys,os
# sys.path.append(os.getcwd())
# from src.database.database import Database


# database = Database()
# database.connect_db()
########################

from src.database.schemas.cameras_schema import Cameras
from src.io_manager.sensors.camera import IP_Camera


class CameraManager:

    def __init__(self) -> None:
        # get all the camera address from db and create a camera
        # instance
        # active_cameras = Cameras.objects(active=True).only('camera_address', 'device_name', 'zones')
        print("#####################################################################################################################################################")
        active_cameras = Cameras.objects(active=True).only('camera_address', 'device_name', 'zones', 'rotation', 'process_skip_frame')
        self.cameras = [IP_Camera(camera_doc.camera_address, camera_doc.device_name, camera_doc.zones , camera_doc.rotation) \
                        for camera_doc in active_cameras]

        active_cameras = Cameras.objects(active=True)
        for cam in active_cameras:
            print(f"Camera {cam.device_name} rotation: {cam.rotation}")
    
    def get_cameras(self):
        return self.cameras
    
    

if __name__ == "__main__":


    camera_manager = CameraManager()
