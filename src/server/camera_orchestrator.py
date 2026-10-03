

import multiprocessing
import uuid
import redis
import time
from src.database.database import Database
from src.database.schemas.camera_config_schema import Cameras
from src.database.schemas.models_schema import Models
from src.server.pipeline_runner import run_camera_pipeline # Import the worker above

class CameraOrchestrator:
    def __init__(self):
        # Redis Connection (Manager side)
        self.redis = redis.Redis(host='localhost', port=6380,password="neosmart@2025", decode_responses=True)
        
        # Database Connection
        self.db = Database()
        self.db.connect_db()

    def start_single_camera(self, camera_id: str, rtsp_url_override: str = None):
        # ... (Redis checks and DB fetching remain the same) ...
        
        # 1. Fetch Camera Document
        camera_doc = Cameras.objects(id=camera_id).first()
        if not camera_doc:
            raise ValueError(f"No Camera found in DB for ID: {camera_id}")

        # 2. Fetch Camera Config
        # camera_config = CameraConfig.objects(cameraOid=camera_id).first()
        # if not camera_config:
        #     raise ValueError(f"No CameraConfig found for Camera ID: {camera_id}")

        # ... (Service and Model fetching logic remains the same) ...
        # service_item = camera_config.services[0]
        
        model_ref = getattr(camera_doc, 'baseModel', None)
        if hasattr(model_ref, 'modelName'):
            model_doc = model_ref
        else:
            model_doc = Models.objects(id=model_ref).first()

        # ---------------------------------------------------------
        # FIX: FORCE USE OF cameraAddress
        # ---------------------------------------------------------
        
        # Priority 1: API Override
        final_rtsp_url = rtsp_url_override 
        
        # Priority 2: 'cameraAddress' from the Camera Document (This is what you want)
        if not final_rtsp_url:
            final_rtsp_url = getattr(camera_doc, 'cameraAddress', None)

        # Priority 3: 'cameraAddress' from the Camera Config (Fallback)
        # if not final_rtsp_url:
        #     final_rtsp_url = camera_config.cameraAddress

        if not final_rtsp_url:
             raise ValueError("No 'cameraAddress' found in Database.")

        # ---------------------------------------------------------
        
        # Construct Payload
        config_payload = {
            "deviceName": camera_doc.deviceName,
            # "zones": [zone.id for zone in service_item.zones],
            "rotation": camera_doc.rotation or 0,
            
            "modelName": model_doc.modelName,
            "modelPath": model_doc.modelPath,
            "trackerPath": model_doc.trackerPath,
            "convertEngine": model_doc.convertEngine,
            "conf": model_doc.conf,
            "classIds": model_doc.classIds,
            "inputDims": model_doc.inputDims,
            
            "security_enabled": True, 
            "plot": True,
            "device": "cuda:0"
        }

        # ... (Process spawning code remains the same) ...
        
        new_run_id = str(uuid.uuid4())
        
        self.redis.hset(f"cam:{camera_id}", mapping={
            "run_id": new_run_id,
            "status": "PROVISIONING",
            "active_url": final_rtsp_url, 
            "last_seen": time.time()
        })

        proc = multiprocessing.Process(
            target=run_camera_pipeline,
            args=(camera_id, final_rtsp_url, config_payload, new_run_id),
            daemon=True
        )
        proc.start()
        
        self.redis.hset(f"cam:{camera_id}", mapping={"pid": proc.pid})

        return {
            "status": "STARTED", 
            "camera_id": camera_id, 
            "run_id": new_run_id,
            "stream_url": final_rtsp_url
        }

    def stop_single_camera(self, camera_id: str):
        # We just update Redis. The Worker Process (Detector) will see the change and stop itself.
        self.redis.hset(f"cam:{camera_id}", mapping={
            "status": "STOPPED", 
            "run_id": "STOPPED"
        })
        return {"status": "STOP_SIGNAL_SENT"}