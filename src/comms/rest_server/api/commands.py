import sys
from classy_fastapi import Routable, get, post
from fastapi.responses import StreamingResponse
from src.orchestrator.orchestrator import Orchestrator
import base64
from pydantic import BaseModel
import time
from src.io_manager.sensors.camera import IP_Camera
from src.comms.rest_server.api.jwt_utils import JWT_token, User
from fastapi import Depends


class Data(BaseModel):
    camera_address: str
    device_name: str
class CameraRoutes(Routable):

    def __init__(self, orchestrator:Orchestrator) -> None:
        super().__init__()
        self.orchestrator = orchestrator


    @get("/camera/start/")
    def start(self):
        """
        This will start the camera live stream in a separate processes
        """
        if self.orchestrator.shared_dict["orchestrator"]["run"] == 0:
            
            # if 1:
            try:

                self.orchestrator.start()

            except Exception as e:
                self.orchestrator.shared_dict["orchestrator"]["run"] = 0
                self.orchestrator.shared_dict["orchestrator"]["error"] = e

        time.sleep(1)

        resp = StreamingResponse( 
                self.orchestrator.stream_orchestrator(), 
                media_type='multipart/x-mixed-replace; boundary=frame'
            )
        
        return resp


    @get("/camera/stop/")
    def stop(self):
        """
        Stop the camera stream and join the processesfalseture
        """
        try:
            self.orchestrator.stop()
        except Exception as e:
            print(f"Exception while stopping : {e}")
            # sys.exit()

        return {"message": 'ip-camera_stream_stopped'}, 200
    
    @post("/camera/connect/")
    def connect(self, data: Data):
        ip_address = data.camera_address
        device_name = data.device_name
        print(f"Connecting to camera: {device_name} at {ip_address}")
        try:
            camera = IP_Camera(ip_address, device_name)   
            frame_bytes = camera.get_frame() 
            jpg_as_text = base64.b64encode(frame_bytes)

            jpg_as_text = jpg_as_text.decode('utf-8')
            
            

        except Exception as e:
            print(f"Exception while connecting to camera: {e}")
            sys.exit() 

        return {"message": 'ip-camera_connected', "frame": jpg_as_text}, 200

