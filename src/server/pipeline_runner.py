import redis
import time
import json
from src.core.sensors.ip_camera import IP_Camera
from src.core.think.machine_learning.computer_vision.detector import Detector 
from src.database.schemas.models_schema import Models
from src.monitoring_stack.mongodb_logger import initialize_logger

def run_camera_pipeline(camera_id, rtsp_url, config_dict, run_id):
    """
    This function runs in a completely separate OS Process.
    """
    # -------------------------------------------------------------
    # FIX IS HERE: Remove 'self.redis'
    # -------------------------------------------------------------
    r = redis.Redis(
        host='localhost', 
        port=6380,
        password="neosmart@2025", # Ensure this matches your Docker setup
        decode_responses=True
    )
    
    # 2. Setup Logger
    logger = initialize_logger(category=f"Worker-{camera_id}")
    logger.info(f"Worker Process Started. RunID: {run_id}")

    # 3. Update Redis Status
    r.hset(f"cam:{camera_id}", mapping={"status": "INITIALIZING"})

    try:
        # 4. Initialize IP Camera
        ip_cam = IP_Camera(
            ip_address=rtsp_url,
            device_name=config_dict.get('deviceName', 'Unknown'),
            rotation=config_dict.get('rotation', 0),
            logger=logger,
            zones=config_dict.get('zones', [])
        )
        
        ip_cam.connect()
        
        if not ip_cam.is_open:
            r.hset(f"cam:{camera_id}", mapping={"status": "ERROR", "error": "Connection Failed"})
            logger.error(f"Failed to connect to RTSP: {rtsp_url}")
            return

        # 5. Inject IP Camera into Config
        config_dict['ip_camera'] = ip_cam
        
        # 6. Initialize Detector
        detector = Detector(config=config_dict, logger=logger)
        
        # 7. Update Status to Running
        r.hset(f"cam:{camera_id}", mapping={"status": "RUNNING"})

        # 8. Start Processing Loop
        detector.process(run_id=run_id, redis_client=r, camera_id=camera_id)

    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")
        r.hset(f"cam:{camera_id}", mapping={"status": "CRASHED", "error": str(e)})
    finally:
        # Final cleanup
        try:
            if 'ip_cam' in locals() and ip_cam:
                ip_cam.close()
        except:
            pass
        logger.info(f"Worker {run_id} finished.")