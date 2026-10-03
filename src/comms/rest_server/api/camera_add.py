from fastapi import Request, HTTPException,Path
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
from datetime import datetime
from classy_fastapi import Routable, post, get, put
from src.orchestrator.orchestrator import Orchestrator
from src.comms.rest_server.api.jwt_utils import JWT_token, User
from fastapi import Depends
from src.database.schemas.heatmap_schema import Heatmap

# Define a model for camera details
class CameraConfig(BaseModel):
    device_name: str = "IP Camera #"
    id: int
    description: str = ""
    settings: dict = { }
# Directory to store JSON files
# Change dir for saving files according to local/ remote settings
# CONFIG_DIR = "src\\configs\\io_manager\\sensors"

# # Ensure directory exists
# os.makedirs(CONFIG_DIR, exist_ok=True)

class add_CameraRoutes(Routable):
    def __init__(self, orchestrator:Orchestrator) -> None:
        super().__init__()
        self.orchestrator = orchestrator


    @post("/camera/add/")
    def add_camera(self, camera_config: CameraConfig, id: int = "#", device_name: str = "IP Camera #", description: str = "&&&&&",
    settings: dict = { "ip_address":"",
        "port_no.":"",
        "channel_no.":"",
        "roi":"",
        "entrance_line":[[0,0],[0,0]]}, current_user: User = Depends(JWT_token.get_current_user)):

        """
        Add camera configuration to the config file
        """
        camera_config.device_name = device_name
        camera_config.id = id
        camera_config.description = description
        camera_config.settings = settings
        # Construct the filename
        filename = os.path.join(CONFIG_DIR, f"ip_c{camera_config.id}.json")
        
        # Check if the file already exists
        if os.path.exists(filename):
            raise HTTPException(status_code=400, detail="Camera configuration with the same ID already exists")
        
        # Save camera configuration to file
        with open(filename, "w") as f:
            json.dump(camera_config.model_dump(), f, indent=4)
        
        return {"message": "Camera configuration saved successfully"}
    
    @get("/camera/ip_c{id}/config/")
    def get_camera_config(self, id: int = Path(..., title="The ID of the camera configuration"), current_user: User = Depends(JWT_token.get_current_user)):
        """
        Get the configuration of a specific camera by id
        """
        # Construct the filename
        filename = os.path.join(CONFIG_DIR, f"ip_c{id}.json")
        
        # Check if the file exists
        if not os.path.exists(filename):
            raise HTTPException(status_code=404, detail="Camera configuration not found")
        
        # Load camera configuration from file
        with open(filename, "r") as f:
            camera_config_data = json.load(f)
        
        return camera_config_data
    
    @put("/update_camera/ip_c{id}")
    def update_camera(self, id: int, camera: CameraConfig, current_user: User = Depends(JWT_token.get_current_user)):
        """
        Update the configuration of a specific camera by id
        """
        camera.id = id
        filename = os.path.join(CONFIG_DIR, f"ip_c{camera.id}.json")

        with open(filename, "w") as f:
            json.dump(camera.model_dump(), f, indent=4)
        return {"message": "Camera configuration updated successfully"}
    
    @post("/camera/heatmap")
    async def heatmap(self, request: Request):
        body = await request.json()
        camera_device_name = body.get('camera_device_name')
        date = body.get('date')
        date_timestamp = datetime.strptime(date,"%d-%m-%Y")
        # dateformat expected: "dd-mm-yyyy"
        
        if not camera_device_name or not date:
            raise HTTPException(status_code=400, detail="Missing image_path or date")
        
        # Filter metadata from camera_device_name and timestamp :
        heatmap_doc = Heatmap.objects(camera_name=camera_device_name, date=date_timestamp).first()
        
        image_path = heatmap_doc.heatmap_path 
        # image_path = image_path.replace("__", "/")
        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            raise HTTPException(status_code=404, detail="File not found")
