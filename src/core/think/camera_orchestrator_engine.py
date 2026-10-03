import gc
import multiprocessing
import json
import time
import signal
import glob, os
from src.utils.logger import Logger
from collections import defaultdict
from src.core.machine import Machine
from src.database.database import Database
from src.core.state_manager import StateManager
from src.core.sensors.ip_camera import IP_Camera
from src.core.think.kpi_manager import KPIManager
from src.monitoring_stack.mongodb_logger import initialize_logger
from src.database.schemas.camera_config_schema import CameraConfig
from src.database.schemas.services_schema import Services
from src.database.schemas.stores_schema import Stores
from src.database.schemas.models_schema import Models


class CameraOrchestratorEngine(Machine):
    """
    This class manages camera pipeline tasks and KPI calculations using multiprocessing.
    It spawns separate processes for each camera pipeline and the KPI Manager.
    """

    def __init__(self, machine_name, state_manager, gpu_memory_per_pipeline_mb=800, auto_restart=True):
        """
        :param machine_name: Name of this orchestrator machine
        :param state_manager: An instance of StateManager
        :param gpu_memory_per_pipeline_mb: Estimated GPU memory usage per pipeline
        :param auto_restart: If True, monitors and restarts failed or hung processes.
        """
        super().__init__(machine_name, state_manager)

        self.gpu_memory_per_pipeline_mb = gpu_memory_per_pipeline_mb
        self.auto_restart = auto_restart
        self.active_processes = []  
        self.active_camera_dict = {}  
        self.restart_count = defaultdict(list)
        self.video_processing = False

        self.database = Database()

        self.kpi_manager_flag = False

        if self.kpi_manager_flag:
            self.kpi_manager = KPIManager(database.db_url, db_name="sentinel", log_dir=self.logger.log_dir)

        self.monitor_thread = None  # Initialize monitor_thread to None
        self.monitor_frq_time = 5
        self.logger = initialize_logger(category="camera_orchestrator_engine")
   
    def process(self, input_data):
        """
        Processes the camera pipelines and starts KPI calculations.
        :param input_data: Dictionary containing active cameras and their configurations.
        """
        self.active_cameras = input_data.get('active_cameras', [])[:]
        print(f"Active cameras: {self.active_cameras}")
        self.current_store = Stores.objects().first().name
        if not self.active_cameras:
            self.logger.warning("No active cameras to process.")
            return {}

        shutdown_event = multiprocessing.Event()

        # Graceful shutdown handler
        def signal_handler(sig, frame):
            self.logger.warning(f"Signal {sig} received! Shutting down...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start camera pipelines
        for camera_data in self.active_cameras:
            camera_id = camera_data.get('_id')

            if not camera_id:
                self.logger.warning(f"Camera document missing '_id': {camera_data}")
                continue
            # service_id = camera_data.get('services')
            model_id = camera_data.get('baseModel')
            

            camera_config = CameraConfig.objects(cameraOid=camera_id).first()

            # print( f"Camera Config for {camera_id}: {camera_config}" )
            # print( f"Services in Camera Config: {camera_config.services if camera_config else 'N/A'}" )

            if not camera_config or not camera_config.services:
                self.logger.warning(f"No configuration or services found for camera: {camera_data.get('deviceName')}")
                continue
            # print
            # Find the FIRST service with a valid pipelinePath and start exactly
            # ONE detector process per camera. Side-services (age, gender, etc.)
            # run as their own Circus watchers and listen to the DB — they do NOT
            # need a separate detector process, so we must not start one for them.
            self.logger.info(camera_config.services)

            pipeline_service_item = None
            for service_item in camera_config.services:
                self.logger.info("service item")
                self.logger.info(service_item)
                svc = service_item.service
                if svc and svc.pipelinePath:
                    pipeline_service_item = service_item
                    break   # ← only one detector per camera

            if pipeline_service_item is None:
                self.logger.warning(f"No service with a pipelinePath found for camera {camera_data.get('deviceName')}; skipping.")
                continue

            service = pipeline_service_item.service
            zones   = [zone.id for zone in pipeline_service_item.zones]

            combined_camera_data = camera_data.copy()
            combined_camera_data['zones']      = zones
            combined_camera_data['service_id'] = service.id
            combined_camera_data['model_id']   = model_id
            combined_camera_data['pipelinePath'] = service.pipelinePath
            path = service.pipelinePath
            proc_name = f"CameraPipeline-{combined_camera_data.get('deviceName', 'unknown')}"

            proc = multiprocessing.Process(
                target=run_camera_pipeline,
                args=(combined_camera_data, path, self.current_store, shutdown_event),
                daemon=False
            )
            proc.start()
            print("started", proc_name)
            self.logger.info(f"Started process '{proc_name}' (PID: {proc.pid})")

            self.active_camera_dict[proc_name] = {'process': proc, 'data': combined_camera_data}
            self.active_processes.append((proc_name, proc))
            time.sleep(1)
        try:
            while not shutdown_event.is_set():
                time.sleep(2)
                for name, entry in list(self.active_camera_dict.items()):
                    proc = entry['process']
                    if not proc.is_alive() and self.should_restart(name):
                        self.logger.warning(f"{name} stopped—restarting...")
                        self.cleanup_process(name, proc)
                        cam = entry['data']
                        path_to_use = cam.get('pipelinePath')
                        
                        if path_to_use:
                            new_name = f"CameraPipeline-{cam.get('deviceName','unknown')}-Main"
                            new_proc = multiprocessing.Process(
                                target=run_camera_pipeline,
                                args=(cam, path_to_use, self.current_store, shutdown_event),
                                daemon=False
                            )
                            new_proc.start()
                            self.logger.info(f"Restarted {new_name}")
                            self.active_camera_dict[new_name] = {'process': new_proc, 'data': cam}
                        else:
                            self.logger.warning(f"Cannot restart camera {cam.get('deviceName')}—no pipelinePath")
        except KeyboardInterrupt:
            self.logger.info("Interrupted.")
        finally:
            self.logger.info("Shutdown received — waiting for camera pipelines to drain remaining frames…")
            for entry in self.active_camera_dict.values():
                p = entry['process']
                if p.is_alive():
                    # Give each camera pipeline up to 1 hour to drain its
                    # frame queue gracefully before forcibly killing it.
                    p.join(timeout=3600)
                    if p.is_alive():
                        self.logger.warning(f"Drain timeout for PID {p.pid} — terminating.")
                        p.terminate()
            self.logger.info("All camera pipelines finished.")
            
    def should_restart(self, name, window=300, max_restarts=5):
        now = time.time()
        self.restart_count[name] = [t for t in self.restart_count[name] if now - t <= window]
        if len(self.restart_count[name]) >= max_restarts:
            self.logger.error(f"{name} flapping: disabling restarts for {window}s")
            return False
        self.restart_count[name].append(now)
        return True

    def cleanup_process(self, name, proc):
        """
        Manually cleans up a process that has stopped.
        :param name: The name of the process to clean up.
        :param proc: The process object to clean up.
        """
        # Terminate the process if it's still alive
        if proc.is_alive():
            proc.terminate()
            proc.join()  # Make sure the process has terminated properly
        
        # Remove the process from active tracking
        if (name, proc) in self.active_processes:
            self.active_processes.remove((name, proc))

        # Remove any associated camera data
        if name in self.active_camera_dict:
            del self.active_camera_dict[name]
        
        gc.collect()
        
        
def run_camera_pipeline( camera_doc, pipeline_config_path, current_store, shutdown_event):
    """
    Executed in a separate process. Sets up the IP camera and runs the detector.
    """
    # --- 1. Get Camera-Specific Details ---
    ip_address = camera_doc.get('cameraAddress')
    device_name = camera_doc.get('deviceName', 'camera-001')
    zones = camera_doc.get('zones', [])
    rotation = camera_doc.get('rotation' , 0)
    process_skip_frame = camera_doc.get('processSkipFrame', 1)
    save_ploted = camera_doc.get('savePlottedFrame', True)
    save_raw_videos = camera_doc.get('saveRawFrame', True)

    # print(f" with zones: {zones}")
    
    logger = initialize_logger(category=device_name)
    database = Database()
    database.connect_db()
    ip_camera = IP_Camera(ip_address=ip_address, device_name=device_name, zones=zones, process_skip_frame=process_skip_frame, rotation=rotation)
    ip_camera.connect()

    # --- 2. Load the Base Config from JSON ---
    with open(pipeline_config_path, 'r') as f:
        pipeline_cfg = json.load(f)
    detector_config = pipeline_cfg["pipeline"]["detector"]["config"]
    module_path = pipeline_cfg["pipeline"]["detector"]["module"]
    class_name = pipeline_cfg["pipeline"]["detector"]["name"]

    # --- 3. Add Camera-Specific Info to the Config ---
    # This fixes the "'Detector' object has no attribute 'deviceName'" error
    detector_config['deviceName'] = device_name
    detector_config['ip_camera'] = ip_camera
    detector_config['zones'] = zones
    # --- 4. Override with Specifics from the Database ---
    model_id = camera_doc.get('model_id')
    if model_id:
        model_doc = Models.objects(id=model_id).first()
        if model_doc and model_doc.modelPath:
            logger.info(f"Found model '{model_doc.modelName}' in database. Overriding static config.")

            # Create a dictionary of settings from the database document
            db_config_overrides = {
                "modelName": model_doc.modelName,
                "modelPath": model_doc.modelPath,
                "modelType": model_doc.modelType,
                "conf": model_doc.conf,
                "classIds": model_doc.classIds,
                "trackerPath": model_doc.trackerPath,
                "convertEngine": model_doc.convertEngine,
                "inputDims": model_doc.inputDims
            }
            # Update the config with the values from the database
            detector_config.update(db_config_overrides)
            print(f"Final detector config: {detector_config}")
        else:
            logger.warning(f"Model with ID '{model_id}' not found in DB. Using config from JSON.")
    else:
        logger.warning("No model_id provided in camera_doc. Using config from JSON.")

    # --- 5. Initialize the Detector with the FINAL, Merged Config ---
    try:
        module = __import__(module_path.replace('/', '.').replace('.py', ''), fromlist=[class_name])
        detector_class = getattr(module, class_name)
    except Exception as e:
        logger.error(f"Error importing detector module: {e}")
        return

    # Add remaining settings
    detector_config["save_ploted"] = save_ploted
    detector_config["save_videos"] = save_raw_videos

    try:
        detector = detector_class(detector_config, logger=logger, shutdown_event=shutdown_event)
    except Exception as e:
        logger.error(f"Detector __init__ failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # --- 6. Run the Detector ---
    try:
        detector.process(run_id=None, redis_client=None, camera_id=camera_doc.get('_id'))
    finally:
        try:
            if hasattr(detector, "metadata_handler") and hasattr(detector.metadata_handler, "close"):
                detector.metadata_handler.close()
        except Exception as e:
            logger.exception(e)
        try:
            ip_camera.disconnect()
        except Exception:
            pass

    def send_data(self, processed_data):
        pass
