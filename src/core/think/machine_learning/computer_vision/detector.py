import cv2
from ultralytics import YOLO
import queue
import threading
import torch
import datetime
import time
import os
import glob
import numpy as np
from collections import deque
from src.utils import plot
from src.database.database import Database
from src.database.metadata_handler import MetadataHandler
from src.database.schemas.zones_schema import Zones
from src.database.schemas.cameras_schema import Cameras
from src.database.schemas.models_schema import Models
from src.core.actuators.frame_handler import FrameHandler
from src.core.think.manager.zone_manager import ZoneManager
from shapely.geometry import Point, Polygon, LineString, box
from src.core.think.manager.predictor_manager import PredictorManager
from src.core.think.machine_learning.computer_vision.safety_tracker import SecurityModule


class Detector:
    """
    Universal Detector that uses UniversalPredictor to detect objects with different models,
    track them, and handle zone/line-based logic.
    """

    def __init__(self, config, logger=None, shutdown_event=None):
        # 1. Store the incoming config and set basic attributes
        self.config = config
        self.device = self.config.get('device', 'cuda:0')
        self.plot = self.config.get('plot', False)
        self.logger = logger
        self.shutdown_event = shutdown_event

        self.database = Database()
        self.database.connect_db()

        # 2. Build the model_config with the CORRECT snake_case keys that PredictorManager expects
        self.model_config = {
            'modelName': self.config.get('modelName'),
            'model_path': self.config.get('modelPath'),         
            'tracker_path': self.config.get('trackerPath'),       
            'convert_engine': self.config.get('convertEngine'),   
            'conf': self.config.get('conf'),
            'class_ids': self.config.get('classIds'),      
            'input_dims': self.config.get('inputDims')        
        }
        
        # 3. Initialize the Predictor, which now receives the correctly formatted config
        self.predictor = PredictorManager(
            model_config=self.model_config,
            device=self.device,
            logger=self.logger,
            plot_overlays=self.config.get("plot_overlays", True),
        )
        self.logger.info(f"Universal Predictor initialized with {self.model_config.get('modelName')} model.")

        self.ip_camera = self.config.get('ip_camera')
        if not self.ip_camera:
            self.logger.error("No 'ip_camera' instance provided in config.")
            raise ValueError("IP Camera configuration is missing.")
        
        # 4. Correctly get deviceName and query the database
        self.device_name = self.config.get('deviceName')
        self.device_id = None

        try:
            camera_doc = Cameras.objects(deviceName=self.device_name).first()
            if camera_doc:
                self.device_id = camera_doc.id
                self.logger.info(f"Found camera in DB. Using device_id: {self.device_id}")
            else:
                self.logger.error(f"Camera '{self.device_name}' not found in DB!")
        except Exception as e:
            self.logger.error(f"Error fetching camera ID from DB: {e}")

        # --- The rest of your file is correct and does not need to be changed ---
        
        self.is_video_file = self.ip_camera.is_video_file
        self.fps = self.ip_camera.fps

        # Initialize Security Module if enabled
        self.security_module = None
        if self.config.get("security_enabled", False):
            _, frame = self.ip_camera.capture.read()
            if frame is not None:
                template_frame = frame
                self.security_module = SecurityModule(self.device_name, template_frame)
                self.logger.info("Security Module Initialized.")
            else:
                self.logger.warning("Could not read frame to initialize Security Module.")

        # Initialize zone management
        self.zones = self.config.get("zones", [])
        print(f"Zones Configured: {self.zones}")
        self.zone_manager = ZoneManager(logger=self.logger, camera={"device_name": self.device_name})
        self.zone_manager.zones = list(self.zones or [])
        self.zone_manager.set_zones()
        
        if self.is_video_file:
            self.video_path = self.ip_camera.ip_address
        else:
            self.video_path = None

        self.rotation = self.ip_camera.rotation
        if self.rotation in [90, 270]:
            rotated_frame_size = (int(self.ip_camera.frame_height), int(self.ip_camera.frame_width))
        else:
            rotated_frame_size = (int(self.ip_camera.frame_width), int(self.ip_camera.frame_height))
        
        self.original_w = rotated_frame_size[0]
        self.original_h = rotated_frame_size[1]

        self.target_fps = 8

        current_datetime = datetime.datetime.now(datetime.timezone.utc)
        self.run_time_str = current_datetime.strftime("%d%m%Y%H%M%S%f")
            
        self.frame_handler = FrameHandler(
            device_name=self.device_name,
            base_dir="./results/frames",
            default_kind="raw",
            image_format="jpg",
            jpeg_quality=60,
            enc_workers=5,
            io_workers=5,
            logger=self.logger,
            max_width=self.config.get("save_max_width", 1280)
        )

        self.metadata_handler = MetadataHandler(
            self.ip_camera.ip_address, 
            self.device_id, 
            logger=self.logger
        )
        
        # Will be initialized in process()
        self.frame_queue = None
        self.last_processed_file = os.path.join(self.frame_handler.base_dir, f"{self.device_name}_last_processed.txt")

        # Record the moment this detector instance was created.
        # Used as a hard cutoff in crash-recovery so we never replay frames
        # that were saved in a previous (older) run.
        self.startup_time = datetime.datetime.now(datetime.timezone.utc)

    def predict(self, frame, current_time, frame_number, raw_path=None):
        # Frame is already rotated and saved in capture_thread

        start_time = time.time()

        track_ids_dict, frame = self.predictor.predict(
            frame=frame,
            current_time=current_time,
            run_time_str=self.run_time_str
        )

        if getattr(self.zone_manager, "zone_dict", None):
            frame = plot.plot_shapes(frame, self.zone_manager.zone_dict)
            
        if self.plot:
            plotted_path = self.frame_handler.submit(
                frame, ts_utc=current_time, frame_number=int(frame_number), kind="plotted"
            )
        else:
            plotted_path = None

    
        end_time = time.time()
        inference_time = end_time - start_time
        
        # [NEW] Log the model inference timing and the queue size to see if Model is too slow!
        qsize = self.frame_queue.qsize() if getattr(self, 'frame_queue', None) else 0
        self.logger.info(f"[{self.device_name}] Processed frame {frame_number} | Inference Time: {inference_time:.4f}s | Queue size: {qsize}")

        _ = self.process_detection_results_thread(
            track_ids_dict, frame, current_time, frame_number, inference_time, raw_path, plotted_path
        )

        if getattr(self, "zone_manager", None) and track_ids_dict:
            track_ids_dict = self.zone_manager.update_track_ids_status(track_ids_dict)
    
        return frame
    
    def rotate_frame(self, frame, angle):
        # This method is correct
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return frame

    def process_detection_results(self, arg_queue):
        # This method is correct
        track_ids_dict, frame, current_time, frame_number, inference_time, raw_path, plotted_path = arg_queue.get()
        arg_queue.task_done()

        if not track_ids_dict:
            track_ids_dict = {}  

        if self.zone_manager:
            track_ids_dict = self.zone_manager.update_track_ids_status(track_ids_dict)
            if self.zone_manager.zone_dict:
                frame = plot.plot_shapes(frame, self.zone_manager.zone_dict)

        # Scale bounding boxes to match the saved raw_frame dimensions
        max_width = getattr(self.frame_handler, "max_width", None)
        orig_height, orig_width = frame.shape[:2]
        
        if max_width and orig_width > max_width:
            scale = max_width / orig_width
            for track_id, info in track_ids_dict.items():
                if isinstance(info, dict) and "bbox" in info:
                    bbox = info["bbox"]
                    info["bbox"] = [
                        int(bbox[0] * scale),
                        int(bbox[1] * scale),
                        int(bbox[2] * scale),
                        int(bbox[3] * scale)
                    ]

        if self.metadata_handler:
            processed_data = {
                "time_stamp": current_time,
                "raw_frame_path": raw_path, 
                "plotted_frame_path": plotted_path,
                "frame_number": frame_number,
                "track_ids_info": track_ids_dict, 
                "inference_time": float(inference_time)
            }         
            self.metadata_handler.process(processed_data)

    def process_detection_results_thread(self, results, frame, current_time, frame_number, inference_time, raw_path, plotted_path):
        # This method is correct
        arg_queue = queue.Queue(maxsize=1)
        arg_queue.put((results, frame, current_time, frame_number, inference_time, raw_path, plotted_path))

        thread = threading.Thread(target=self.process_detection_results, args=(arg_queue,))
        thread.daemon = True
        thread.start()
        return thread
    
    # In src/core/think/machine_learning/computer_vision/detector.py

    # Update the process signature to accept run_id and redis_client
    def process(self, run_id=None, redis_client=None, camera_id=None):
        self.logger.info(f"Entering live loop: is_open={self.ip_camera.is_open}, fps={self.ip_camera.fps}")
        
        if not self.ip_camera.is_open:
            self.logger.error("IP camera is not open; exiting.")
            return

        interval = datetime.timedelta(seconds=1 / self.target_fps)
        self.no_frame_count = 0
        self.cam_result = True
        
        # --- [NEW] Decoupled Capture Thread using Disk Backing ---
        self.frame_queue = queue.Queue(maxsize=0)
        self.last_raw_path = ""
        self.last_processed_time = None  # UTC datetime of the last processed frame
        
        if os.path.exists(self.last_processed_file):
            with open(self.last_processed_file, "r") as f:
                self.last_raw_path = f.read().strip()
            # Parse the timestamp from the stored path so we can compare by time,
            # not by alphabetical string (which breaks across session restarts).
            try:
                _lp_basename = os.path.basename(self.last_raw_path).replace(".jpg", "")
                _lp_parts = _lp_basename.rsplit("_f", 1)
                if len(_lp_parts) == 2:
                    self.last_processed_time = datetime.datetime.strptime(
                        _lp_parts[0], "%Y-%m-%d_%H-%M-%S_%f"
                    ).replace(tzinfo=datetime.timezone.utc)
                    self.logger.info(f"Resuming from last processed time: {self.last_processed_time}")
            except Exception as _e:
                self.logger.warning(f"Could not parse last_processed_time from '{self.last_raw_path}': {_e}")
                
        # --- Recovery logic: Enqueue unprocessed frames from disk ---
        try:
            raw_dirs = sorted(glob.glob(os.path.join(self.frame_handler.base_dir, "*", self.device_name, "raw")))
            # Only check the latest 2 days to avoid long startup
            for raw_dir in raw_dirs[-2:]:
                frames = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
                for frame_path in frames:
                    # Parse timestamp and frame number from filename
                    basename = os.path.basename(frame_path).replace(".jpg", "")
                    parts = basename.rsplit("_f", 1)
                    if len(parts) != 2:
                        continue

                    frame_num = int(parts[1])
                    ts_str = parts[0]
                    try:
                        parsed_time = datetime.datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S_%f").replace(tzinfo=datetime.timezone.utc)
                    except Exception:
                        parsed_time = datetime.datetime.now(datetime.timezone.utc)

                    # --- Skip already-processed frames (timestamp-based) ---
                    # Compare using parsed UTC timestamps, not filename strings.
                    # This is correct even across session restarts where frame
                    # numbers reset to 0 but timestamps keep advancing.
                    if self.last_processed_time and parsed_time <= self.last_processed_time:
                        continue

                    # --- STARTUP CUTOFF (time-based, 48-hour window) ---
                    # Skip frames that are more than 48 hours old to avoid
                    # replaying very stale data. Use a UTC time window (not an
                    # IST date comparison) so that a restart just after midnight
                    # IST does NOT accidentally drop same-UTC-day frames.
                    cutoff_time = self.startup_time - datetime.timedelta(hours=48)
                    if parsed_time < cutoff_time:
                        continue

                    self.frame_queue.put({
                        "raw_path": frame_path,
                        "time": parsed_time,
                        "frame_number": frame_num
                    })
            self.logger.info(f"Recovered {self.frame_queue.qsize()} unprocessed frames from disk.")
        except Exception as e:
            self.logger.error(f"Error recovering unprocessed frames: {e}")

        # The global frame num must continue to increase
        global_frame_num = 0
        
        def capture_thread():
            nonlocal global_frame_num
            interval = datetime.timedelta(seconds=1 / self.target_fps)
            last_enqueue_time = None
            
            while self.ip_camera.is_open and not (self.shutdown_event and self.shutdown_event.is_set()):
                result, frame = self.ip_camera.capture.read()
                if not result or frame is None:
                    time.sleep(0.01)
                    continue
                
                current_time = datetime.datetime.now(datetime.timezone.utc)
                if last_enqueue_time is None or (current_time - last_enqueue_time) >= interval:
                    
                    if self.rotation != 0:
                        frame = self.rotate_frame(frame, self.rotation)

                    # Submit frame to write perfectly in parallel
                    raw_path = self.frame_handler.submit(
                        frame, ts_utc=current_time, frame_number=global_frame_num, kind="raw"
                    )
                    
                    self.frame_queue.put({
                        "raw_path": raw_path,
                        "time": current_time,
                        "frame_number": global_frame_num
                    })
                    global_frame_num += 1
                    last_enqueue_time = current_time
        
        # Start capture thread
        cap_t = threading.Thread(target=capture_thread, daemon=True)
        cap_t.start()
        
        try:
            suicide_counter = 0
            shutting_down = False
            
            while self.ip_camera.is_open:
                
                # --- 0. GRACEFUL DRAIN CHECK ---
                # Once shutdown is requested, stop accepting new camera frames
                # (capture_thread will exit on its own) but keep processing until
                # the queue is completely empty so no frames are left behind.
                if self.shutdown_event and self.shutdown_event.is_set():
                    if not shutting_down:
                        shutting_down = True
                        self.logger.info(
                            f"Shutdown requested. Draining remaining "
                            f"{self.frame_queue.qsize()} frame(s) before stopping..."
                        )
                    if self.frame_queue.empty():
                        self.logger.info("Frame queue fully drained. Stopping cleanly.")
                        break

                # --- 1. THE SUICIDE CHECK ---
                # Check if this process is still the valid one in Redis
                if redis_client and camera_id and run_id:
                    # We check Redis every ~10 frames or 2 seconds to save bandwidth
                    if suicide_counter % 10 == 0:
                        active_run_id = redis_client.hget(f"cam:{camera_id}", "run_id")
                        if active_run_id != run_id:
                            self.logger.warning(f"[STOP] RunID Mismatch. Mine: {run_id}, DB: {active_run_id}. Exiting.")
                            break
                        
                        # Send Heartbeat
                        redis_client.hset(f"cam:{camera_id}", mapping={"last_seen": time.time()})
                suicide_counter += 1

                # --- 2. Consume Frame ---
                try:
                    # Wait up to a short time for a frame path
                    frame_data = self.frame_queue.get(timeout=2.0)
                    frame_ts = frame_data["time"]
                    frame_num = frame_data["frame_number"]
                    raw_path = frame_data["raw_path"]
                    self.no_frame_count = 0  # Reset counter
                except queue.Empty:
                    self.no_frame_count += 1
                    # During graceful drain, an empty queue just means we're done
                    if shutting_down:
                        self.logger.info("Queue drained after shutdown. Stopping.")
                        break
                    if self.no_frame_count >= 10:
                        self.logger.warning("Too many empty frames/timeout. Exiting.")
                        break
                    continue

                # Wait for file to exist (if it was just launched by FrameHandler)
                wait_time = 0
                while not os.path.exists(raw_path) and wait_time < 5.0:
                    time.sleep(0.05)
                    wait_time += 0.05
                    
                frame = cv2.imread(raw_path)
                if frame is None:
                    continue
                
                if frame.shape[1] != self.original_w or frame.shape[0] != self.original_h:
                    frame = cv2.resize(frame, (self.original_w, self.original_h))
                    
                raw_frame = frame.copy()
                processed_frame = self.predict(raw_frame, frame_ts, frame_num, raw_path)
                
                # Update last processed state
                try:
                    with open(self.last_processed_file, "w") as f:
                        f.write(raw_path)
                except Exception as e:
                    self.logger.error(f"Failed to write last_processed_file: {e}")
                
                if self.security_module:
                    self.security_module.process_frame(raw_frame)

        except Exception as e:
            self.logger.exception(f"Critical error in Detector live loop for {self.device_name}: {e}")
        finally:
            self.logger.info(f"🧹 Cleaning up resources for {self.device_name}")
            
            if hasattr(self, 'predictor'):
                self.predictor.finish()

            if self.ip_camera.capture:
                self.ip_camera.capture.release()

            if hasattr(self, "frame_handler") and self.frame_handler:
                self.frame_handler.close()

            if hasattr(self, 'metadata_handler'):
                self.metadata_handler.close()
            
            self.logger.info("Released VideoCapture object.")
            self.logger.info("Cleaned up all resources.")
            print("************************* Completed *************************\n")