
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cryptography")

# main.py
import multiprocessing as mp
import multiprocessing.process
import os

# Force CPU mode globally by hiding CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import logging
import subprocess
import multiprocessing
import signal
import threading
import time
import datetime

from src.core.state_manager import StateManager  
from src.utils.logger import Logger
from src.monitoring_stack.mongodb_logger import initialize_logger
from src.process_supervisor.circus_embed import EmbeddedCircus



_shutdown = threading.Event()
def _handle_signal(sig, frame):
    _shutdown.set()

def setup_logging():
    # Get current date-time for the log folder
    date_time = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/{date_time}"  # Base log folder path
    os.makedirs(log_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Define the logging format
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    # Create a custom log handler to write logs into the new directory

    logging.info(f"Log directory created: {log_dir}")
    return log_dir

def start_metrics_collector(logger):
    try:
        metrics_process = multiprocessing.Process(
            target=lambda: subprocess.run(
                ["python3", "src/monitoring_stack/integration_grafana.py"],
                env={**os.environ, "PYTHONPATH": os.getcwd()}
            )
        )
        metrics_process.daemon = True
        metrics_process.start()
        logger.info("prometheus metrics collector started on port 9990")
    except Exception as e:
        logger.exception("failed to start metrics: %s", e)



def main():

    log_dir = setup_logging()  # Setup logging
    logging.info("Logging initialized")

    logger = initialize_logger(category="mainpipeline")
    logger.info("new logger established")
    logger.info("logs from bealpur")

    circus = EmbeddedCircus()
    circus.start(log_dir=log_dir)

    # Install signal handlers and wait here until interrupted
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not _shutdown.is_set():
            time.sleep(0.5)
    finally:
        try:
            circus.stop()
        except Exception:
            logger.exception("Error stopping embedded Circus")


if __name__ == '__main__':
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()




# import uvicorn
# from fastapi import FastAPI
# from src.server.camera_orchestrator import CameraOrchestrator
# from pydantic import BaseModel

# app = FastAPI()
# orchestrator = CameraOrchestrator()

# class CameraReq(BaseModel):
#     camera_id: str

# @app.post("/camera/start")
# def start_cam(req: CameraReq):
#     return orchestrator.start_single_camera(req.camera_id)

# @app.post("/camera/stop")
# def stop_cam(req: CameraReq):
#     return orchestrator.stop_single_camera(req.camera_id)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8765)
