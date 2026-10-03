import queue
import threading
import cv2
from ultralytics import YOLO
import torch
import datetime
import numpy as np
import os, glob
from shapely.geometry import Point, Polygon, LineString, box
from collections import deque, defaultdict
from src.utils import plot
from concurrent.futures import ThreadPoolExecutor
from src.database.schemas.sentinel_poc_schema import Cameras, Zones
from src.database.database import Database
from src.database.metadata_handler import MetadataHandler
from src.core.actuators.video_writer_manager import VideoWriterManager
from src.core.actuators.video_writer_h265 import VideoWriterManager_h265


from src.core.think.machine_learning.computer_vision.safety_tracker import SecurityModule
from src.core.think.machine_learning.computer_vision.human_profiler import EntryExitHandler

class Detector:
    """
    HumanDetector machine that uses a YOLO model to detect humans in frames,
    track them, and handle zone/line-based logic. Now supports entry/exit events
    with crop collection and callback.
    """

    def __init__(
        self,
        config,
        logger=None,
        collect_frames=5,
        shutdown_event=None):

        self.device = config.get('device', 'cuda:0')
        self.model_path = config.get('model_path')
        self.tracker_path = config.get('tracker_path')
        self.plot = config.get('plot', False)
        self.conf = config.get('conf', 0.6)
        self.class_ids = config.get('class_ids', [0])  # Default to 'person' class
        self.logger = logger
        self.collect_frames = collect_frames
        self.zone_entry_state = {}               # last known region per track
        self.track_history = defaultdict(lambda: deque(maxlen=self.collect_frames))
        self.collecting = set()  
        self.ip_camera = config.get('ip_camera')
        if not self.ip_camera:
            self.logger.error("No 'ip_camera' instance provided in config.")
            raise ValueError("IP Camera configuration is missing.")
# track IDs being collected
        self.shutdown_event = shutdown_event


        self.device_name = self.ip_camera.device_name
        print(f"self.device_name: {self.device_name}")
        self.is_video_file = self.ip_camera.is_video_file
        self.fps = self.ip_camera.fps

        # self.uploader = AzureDatedImageUploader()
        # Initialize database connection
        self.database = Database()
        self.database.connect_db()
        self.entry_exit_handler = EntryExitHandler()
        
        # Sanity checks
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.tracker_path):
            raise FileNotFoundError(f"Tracker file not found at {self.tracker_path}")
        if not torch.cuda.is_available() and self.device.startswith('cuda'):
            self.logger.info("CUDA not available, switching to CPU.")
            self.device = 'cpu'
            
        self.security_module = None
        if config.get("security_enabeled", False):
            self.logger.info("Initializing Security Module...")
            # _, frame = self.ip_camera.capture.read()
            template_frame = self.ip_camera.template_frame  # Load initial template frame
            self.security_module = SecurityModule(self.device_name,template_frame)
            self.logger.info("Security Module Initialized.")


        # Load YOLO model
        try:
            self.logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path, task="detect")
            self.logger.info("YOLO model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise
        self.model.tracker = self.tracker_path
        # self.embedder = ResNetAttentionEmbedder(model_path="src/models/embedding_extractor.pth")
        # IP Camera configs
        # self.ip_camera = config.get('ip_camera')
        # if not self.ip_camera:
        #     self.logger.error("No 'ip_camera' instance provided in config.")
        #     raise ValueError("IP Camera configuration is missing.")

        
        
        # (keep existing initialization of YOLO, ip_camera, zones, video writers, etc.)
        self.zones = self.ip_camera.zones
        self.zone_dict = {}
        self._set_zones()
        self.conf_threshold = config.get('conf_threshold', 0.4)
        
        # Video path
        if self.is_video_file:
            self.video_path = self.ip_camera.ip_address  # Local video file path
        else:
            self.video_path = None  # Live feed

        # Video writing if needed
        current_datetime = datetime.datetime.now(datetime.timezone.utc)
        self.run_time_str = current_datetime.strftime("%d%m%Y%H%M%S%f")
        date = current_datetime.date()
        self.save_video = False
        self.save_raw_video = True

        self.target_fps = 10

        self.raw_video_writer = None
        (frame_width, frame_height) = (self.ip_camera.frame_width, self.ip_camera.frame_height)

        if self.save_raw_video:
            raw_video_dir = os.path.join(f"./results/raw_videos/Xperia/{date}", self.device_name)
            os.makedirs(raw_video_dir, exist_ok=True)

            self.raw_video_writer = VideoWriterManager(
                camera_id=self.ip_camera.device_name,
                output_dir_template=raw_video_dir,
                frame_width=frame_width,
                frame_height=frame_height,
                target_fps=self.target_fps,
                source_fps_val=self.ip_camera.fps,
                original_stream_width=frame_width,
                original_stream_height=frame_height,
                parent_logger=self.logger
            )

            # self.raw_video_writer = VideoWriterManager_h265(
            #     save_videos_flag=self.save_raw_video,
            #     results_base_dir=raw_video_dir,
            #     date=date,
            #     frame_size=(frame_width, frame_height),
            #     fps= self.target_fps,  # Using the target_fps you specified in your original attempt
            #     logger=self.logger
            # )
        

        # self.video_writer = None
        # if self.save_video:
        #     video_dir = os.path.join(f"./results/plotted_videos_benchmark/{date}", self.device_name)
        #     os.makedirs(video_dir, exist_ok=True)

        #     self.video_writer = VideoWriterManager(
        #         camera_id=self.device_name,
        #         output_dir_template=video_dir,
        #         frame_width=self.ip_camera.frame_width,
        #         frame_height=self.ip_camera.frame_height,
        #         target_fps=5,
        #         source_fps_val=self.ip_camera.fps,
        #         original_stream_width=self.ip_camera.frame_width,
        #         original_stream_height=self.ip_camera.frame_height,
        #         parent_logger=self.logger
        #     )
            
        self.metadata_handler = MetadataHandler(self.ip_camera.ip_address, self.ip_camera.device_name, logger=self.logger)


        self.store_crops_flag = True
        if self.store_crops_flag:
            self.crop_dir = "./results/track_ids/Xperia/"

    def _get_center(self, bbox):
        """
        Calculate center of the bounding box [x1, y1, x2, y2].
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[3])
        return (x_center, y_center)

    def _set_zones(self):
        """
        Convert zone definitions to Shapely geometries and store in self.zone_dict,
        skipping any invalid shapes.
        """
        self.zone_dict = {}
        if not self.zones:
            self.logger.warning("No zones found for this camera.")
            return

        # Fetch Zone details from MongoDB using the stored ObjectId references
        zone_objects = Zones.objects(id__in=self.zones)  # Retrieve all Zone documents

        # Store them as a dictionary with name as key and zone_id as value
        self.zone_data = {zone.name: str(zone.id) for zone in zone_objects}
        for zone_name, zone_id in self.zone_data.items():
            zone_doc = Zones.objects(id=zone_id).first()
            if not zone_doc:
                self.logger.warning(f"Zone doc with ID '{zone_id}' not found.")
                continue

            vertices = zone_doc.roi  # e.g. [[x1, y1, x2, y2, ...]]
            if not vertices:
                self.logger.warning(f"No ROI found for zone '{zone_name}'.")
                continue

            # Flatten
            vertices_flat = [int(x) for x in vertices[0]]
            if len(vertices_flat) % 2 != 0:
                self.logger.warning(f"Odd # of coords for zone '{zone_name}'. Skipping.")
                continue

            # Convert to (x, y) pairs
            vertices_points = [
                tuple(vertices_flat[i:i+2]) for i in range(0, len(vertices_flat), 2)
            ]

            shape = None
            shape_type = None

            if len(vertices_points) > 2:
                # Try polygon
                poly = Polygon(vertices_points)
                if poly.is_empty or not poly.is_valid:_
                    self.logger.warning(f"Invalid polygon for zone '{zone_name}'. Skipping.")
                    continue
                shape = poly
                shape_type = "zone"
            elif len(vertices_points) == 2:
                # Try line
                line = LineString(vertices_points)
                # Check if line is actually valid (distinct points)
                if line.is_empty or line.length == 0:
                    self.logger.warning(f"Invalid line (zero length) for zone '{zone_name}'. Skipping.")
                    continue
                shape = line
                shape_type = "line"
            else:
                # Not enough points => skip
                self.logger.warning(f"Insufficient points for zone '{zone_name}'. Skipping.")
                continue

            self.zone_dict[zone_name] = {
                "shape": shape,
                "type": shape_type
            }

        self.logger.debug(f"zone_dict: {self.zone_dict}")

    def _calculate_iou(self, bbox, zone):
        """
        Calculate the Intersection over Union (IoU) between a bounding box and a zone (polygon).
        """
        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            self.logger.error(f"Invalid bbox format in _calculate_iou(): {bbox}")
            return 0

        try:
            x1, y1, x2, y2 = bbox[:4]
            bbox_polygon = box(x1, y1, x2, y2) 
            if not isinstance(zone, Polygon) or zone.is_empty or not zone.is_valid:
                self.logger.error(f"Zone is not a valid polygon: {zone}")
                return 0

            intersection_area = bbox_polygon.intersection(zone).area
            union_area = bbox_polygon.area
            iou_value = intersection_area / union_area if union_area > 0 else 0

            return round(iou_value, 4)  # Return IoU rounded to 4 decimal places

        except Exception as e:
            self.logger.error(f"Error in _calculate_iou: {e}", exc_info=True)
            return 0



    def _check_spatial_relationship_iou(self, shape_info, bbox):
        """
        Determine the IoU of a bounding box with a given zone.
        """
        shape = shape_info["shape"]
        shape_type = shape_info["type"]
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist() 
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            self.logger.warning(f"Skipping invalid bbox in _check_spatial_relationship: {bbox}")
            return 0  

        if shape_type == "zone" and isinstance(shape, Polygon) and shape.is_valid:
            return self._calculate_iou(bbox, shape)

        self.logger.warning(f"Skipping IoU calculation for non-zone shape: {shape_type}")
        
        return 0  
    
    def _check_spatial_relationship_location(self, shape_info, point):
        """
        Determine the relationship of a point to a shape (zone or line).

        - If 'zone' (Polygon): returns {"inside", "outside", "on"}.
        - If 'line' (LineString): 
            - Decide if line is more horizontal or more vertical.
            - For more horizontal: return "above", "below", or "on".
            - For more vertical: return "left", "right", or "on".
        """
        shape = shape_info["shape"]
        shape_type = shape_info["type"]

        # ZONE (polygon)
        if shape_type == "zone":
            if shape.contains(point):
                return "inside"
            elif shape.touches(point):
                return "on"
            else:
                return "outside"   
        elif shape_type == "line":
            line_coords = list(shape.coords)
            if len(line_coords) != 2:
                return "on"

            (x1, y1), (x2, y2) = line_coords
            px, py = point.x, point.y

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) > abs(dy):
                
                cross_val = dx * (py - y1) - dy * (px - x1)
                if abs(cross_val) < 1e-6:
                    return "on"
                elif cross_val > 0:
                    return "above"
                else:
                    return "below"
            else:
                cross_val = dx * (py - y1) - dy * (px - x1)
                if abs(cross_val) < 1e-6:
                    return "on"
                elif cross_val > 0:
                    return "left"
                else:
                    return "right"

        return "unknown"
    

    def _get_lower_midpoint(self, bbox):
        """
        Calculate the lower middle point of the bounding box [x1, y1, x2, y2].
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = bbox[3]  
        return (x_center, y_center)

    def set_track_ids_status(self, track_ids_dict):
        """
        1) Store/update each track_id in self.track_ids_record with a 25-length deque of records.
        2) Determine where the bbox is with respect to each zone.
        3) Compute IoU using the full bounding box.
        4) Remove old track_ids (>5 min without updates).
        """
        for track_id, obj_info in track_ids_dict.items():
            bbox = obj_info["bbox"] 
            instance_dict = {}
            for zone_name, zone_info in self.zone_dict.items():
                iou_value = self._check_spatial_relationship_iou(zone_info, bbox) 
                center_pt = Point(self._get_center(bbox))
                location_status = self._check_spatial_relationship_location(zone_info, center_pt)
                if zone_name not in instance_dict:
                    instance_dict[zone_name] = {}
                instance_dict[zone_name]["iou"] = iou_value
                instance_dict[zone_name]["location"] = location_status
            obj_info["instance_dict"] = instance_dict
        return track_ids_dict

    def predict(self, frame, current_time, frame_number):

        self.raw_video_writer.write_frame(frame)
        # self.raw_video_writer.write_frame(frame,current_time)
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device
        )

        torch.cuda.empty_cache()

        _ = self.process_detection_results_thread(results, frame, current_time, frame_number)

        return frame

    # Example of how to call this function in a separate thread
    def process_detection_results_thread(self, results, frame, current_time, frame_number):
        """
        Creates a temporary queue to pass arguments to the target method
        and starts a new thread for processing.
        """
        # Create a temporary, thread-safe queue for this specific task's data
        arg_queue = queue.Queue(maxsize=1) # maxsize=1 as it's for one item

        # Put the arguments into the queue as a single item (e.g., a tuple)
        arg_queue.put((results, frame, current_time, frame_number))

        # Create the thread, passing the queue to the target method
        # The target method 'self.process_detection_results' will now expect a queue
        thread = threading.Thread(target=self.process_detection_results, args=(arg_queue,))
        thread.daemon = True
        thread.start()
        return thread

    def process_detection_results(self, arg_queue):
        """
        Processes detection results obtained from the provided queue.
        """
        # Get the arguments from the queue
        # This call will block until an item is available in the queue
        results, frame, current_time, frame_number = arg_queue.get()

        # Mark the task as done (important if you were to .join() the queue later, though not critical for this single-item use)
        arg_queue.task_done()

        # Now you can use results, frame, current_time, frame_number as before
        # print(f"Thread {threading.current_thread().name} processing frame {frame_number} at {current_time}")
        # Add your actual detection results processing logic here

        # save raw frame

        


        track_ids_dict = {}
        if not results or len(results) == 0 or results[0].boxes.id is None:
            return track_ids_dict, frame

        boxes = results[0].boxes.xyxy.int().cpu().numpy()      # [N,4]
        track_ids = results[0].boxes.id.int().cpu().numpy()    # [N]
        labels = results[0].boxes.cls.int().cpu().numpy()      # [N]
        confidence = results[0].boxes.conf.cpu().numpy()       # [N]
        class_names = results[0].names
        
        # Fill track_ids_dict
        for idx, track_id in enumerate(track_ids):
            # We'll embed run_time_str into the ID if desired, or keep track_id as is
            updated_track_id = (f"{self.run_time_str}_{track_id:05d}")
            # Convert label to string ("customer" or "employee")
            if labels[idx] == 0:
                label_name = "customer"
            elif labels[idx] == 1:
                label_name = "employee"
            else:
                label_name = class_names[labels[idx]]  
            track_ids_dict[updated_track_id] = {
                "track_id": updated_track_id,
                "bbox": boxes[idx],
                "label": labels[idx],
                "confidence": confidence[idx],
                "label_name": class_names[labels[idx]],
                "current_time": current_time
            }

        # Key call: set track ID statuses
        if self.zone_dict:
            track_ids_dict = self.set_track_ids_status(track_ids_dict)

        #store crops 
        if self.store_crops_flag:
            track_ids_dict = self.store_crops(frame, track_ids_dict)


        self._handle_entry_events(track_ids_dict, frame)

                        # Optionally draw zone shapes for debugging
        if self.zone_dict:
            frame = plot.plot_shapes(frame, self.zone_dict)

                # If self.plot is True, draw the bounding boxes
        if self.plot:
            frame = results[0].plot(conf=False, labels=True, boxes=True, masks=False)



        if track_ids_dict:
            processed_data = {
                "time_stamp": current_time,
                "evidence_path": self.raw_video_writer.get_output_path(),
                "frame_number": frame_number,
                "evidence_frame_number": self.raw_video_writer.frame_number,
                "track_ids_info": track_ids_dict
            }


            self.metadata_handler.process(processed_data)

    def store_crops(self, frame, track_ids_dict):
        successful = 0
        
        for track_id, obj_info in track_ids_dict.items():
            bbox = obj_info["bbox"]  # Use the full bounding box
            label = obj_info["label_name"]  # Use the full bounding box
            track_id_path_list = []

            # … your existing padding + bbox logic …
            x1, y1, x2, y2 = map(int, bbox)
            P = 10
            h, w = frame.shape[:2]
            x1p, y1p = max(0, x1 - P), max(0, y1 - P)
            x2p, y2p = min(w, x2 + P), min(h, y2 + P)
            if x2p <= x1p or y2p <= y1p:
                continue

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            current_datetime = datetime.datetime.now(datetime.timezone.utc) # Store milliseconds
            date = str(current_datetime.date())



            current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            instance_dict = obj_info["instance_dict"]

            if len(instance_dict) > 0:

                not_saved = True
                for zone_name_index in instance_dict:

                    if instance_dict[zone_name_index]['location'] == "inside":
                        zone_name  = zone_name_index.replace(" ", "_")
                        file_name = f"{current_time}-{label}-{instance_dict[zone_name_index]['location']}-{zone_name}.jpg"
                        track_id_dir = os.path.join(self.crop_dir, date, self.device_name, zone_name, track_id)

                        if not os.path.exists(track_id_dir):

                            os.makedirs(track_id_dir, exist_ok=True)

                        out_fp = os.path.join(track_id_dir, file_name)
                        
                        if cv2.imwrite(str(out_fp), crop):
                            track_id_path_list.append(out_fp)
                            not_saved = False

                        # successful += 1
                        # self.logger.info(f"Storing crop for track_id: {track_id} in zone: {zone_name} at path: {out_fp}")

            if not_saved:
                zone_name = "no-zone"
                file_name = f"{current_time}-{label}-_-{zone_name}.jpg"
                track_id_dir = os.path.join(self.crop_dir, date, self.device_name, zone_name, track_id)

                if not os.path.exists(track_id_dir):

                    os.makedirs(track_id_dir, exist_ok=True)


                out_fp = os.path.join(track_id_dir, file_name)
                
                if cv2.imwrite(str(out_fp), crop):
                    track_id_path_list.append(out_fp)
                    not_saved = False
                    # successful += 1

            track_ids_dict[track_id]["track_id_path_list"] = track_id_path_list

        return track_ids_dict
    
    def _handle_entry_events(self, track_ids_dict, frame):
        """
        On outside->inside: collect crops and call entry handler.
        Now also collects detection confidences and passes them.
        """
        for tid, info in track_ids_dict.items():
            inst = info['instance_dict']
            # determine current region
            curr = None
            if inst.get('inside', {}).get('location') in ('inside','on'):
                curr = 'inside'
            elif inst.get('outside', {}).get('location') in ('inside','on'):
                curr = 'outside'
            if curr is None:
                continue

            prev = self.zone_entry_state.get(tid)
            if prev is None:
                # First sighting
                self.zone_entry_state[tid] = curr
                continue

            # Entry event detected
            if prev == 'outside' and curr == 'inside' and tid not in self.collecting:
                self.collecting.add(tid)

            # If collecting, gather frames
            if tid in self.collecting:
                if curr == 'inside':
                    x1, y1, x2, y2 = info['bbox']
                    crop = frame[y1:y2, x1:x2].copy()
                    self.track_history[tid].append({'metadata': info, 'crop': crop})

                # Once enough frames collected, fire callback
                if len(self.track_history[tid]) >= self.collect_frames:
                    batch = list(self.track_history[tid])
                    metadata_list = [e['metadata'] for e in batch]
                    crops_list    = [e['crop']     for e in batch]
                    # Extract YOLO confidences from metadata
                    confidences = [m.get('confidence', 0.0) for m in metadata_list]
                    label_name  = metadata_list[0].get('label_name')
                    # Call entry handler with confidences
                    print(f"Entry event detected for track ID {tid} with {len(crops_list)} crops.")
                    # self.executor.submit(self.entry_exit_handler.process, tid, crops_list, confidences, label_name)
                    self.entry_exit_handler.process(tid, crops_list, confidences, label_name)

                    # Cleanup for this track
                    self.collecting.remove(tid)
                    del self.track_history[tid]

            # Update last known region
            self.zone_entry_state[tid] = curr


    # process() remains as before, calling predict() and handling security_module, etc.

    def process_video(self):
        """
        Main loop: read frames, detect humans, record metadata, optionally stream.
        """

        work_dir = "/home/sentinel/Projects/ambient-machine/outputs/home/sentinel/Projects/ambient-machine/results/videos/Reliance-Digital-Belapur/2025-05-05"

        videos_list = glob.glob(f"{work_dir}/{self.device_name}/*")

        def get_timestamp_from_filename(filepath):
            """Extracts the datetime object from the video filename."""
            filename = os.path.basename(filepath)  # e.g., video_2025-05-05_12-12-55.mp4
            timestamp_str = filename.split('_')[1] + "_" + filename.split('_')[2].split('.')[0] # e.g., 2025-05-05_12-12-55
            return datetime.datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')

        # Sort the list using the extracted timestamp
        videos_list.sort(key=get_timestamp_from_filename)

        descriptor = f"Processing {self.device_name} Video"
        self.logger.info(descriptor)

        video_doc = Cameras.objects(camera_address=self.video_path).first()
        
        frame_number = 0
        initial_time_stamp = datetime.datetime.now(datetime.timezone.utc)
        self.no_frame_count = 0
        self.cam_result = True



        try:
        # if 1:
            total_videos = len(videos_list)

            total_videos = len(videos_list)
            video_count = 0
            video_file = None
            last_video_file = None
            for video_file in videos_list:

                if not os.path.exists(video_file):
                    print(f"path not found: ", video_file)

                if video_file is not None and last_video_file is not None and last_video_file != video_file:
                    self.raw_video_writer.finish_clip(status="done")
                    if self.ip_camera.capture and self.ip_camera.capture.isOpened():
                        self.logger.info(f"Releasing camera capture for {self.device_name}")
                        self.ip_camera.capture.release()
                    self.logger.info(f"finishing videso file {last_video_file} {video_count}/{total_videos}")
                    video_count +=1

                self.ip_camera.ip_address = video_file
                try:
                    self.ip_camera.connect()
                    last_video_file = video_file
                except Exception as e:
                    self.logger.warning(f"Unable to connect to : {video_file}")
                self.cam_result = True

                # print("************************* Processing *************************\n")
                while self.ip_camera.is_open and self.cam_result:
                    if self.shutdown_event and self.shutdown_event.is_set():
                        self.logger.info(f"Detector {self.device_name} (video) received shutdown. Exiting.")
                        break
                    self.cam_result, frame = self.ip_camera.capture.read()

                    if frame is None:
                        print("No frame captured, skipping...")
                        self.no_frame_count += 1
                        if self.no_frame_count >=10:
                            break
                        else:
                            continue

                
                    
                    frame_number += 1
                    current_time = datetime.datetime.now(datetime.timezone.utc)

                    if frame_number % self.ip_camera.process_skip_frame != 0:
                        continue

                    if frame is not None and self.cam_result:
                        
                        raw_frame = frame.copy()
                        
                        processed_frame = self.predict(raw_frame, current_time, frame_number)
                        # print(f"track_ids_dict: {track_ids_dict}")
                        if self.security_module:
                            self.security_module.process_frame(frame)
                        # if self.save_video:
                        #     self.video_writer.write_frame(processed_frame, current_time)


                        # if self.save_video and self.video_writer:
                        #     try:
                        #         # Ensure the processed frame has the correct dimensions
                        #         expected_height = self.ip_camera.frame_height
                        #         expected_width = self.ip_camera.frame_width
                                
                        #         if processed_frame.shape[:2] != (expected_height, expected_width):
                        #             self.logger.warning(f"Resizing processed frame from {processed_frame.shape[:2]} to ({expected_height}, {expected_width})")
                        #             processed_frame = cv2.resize(processed_frame, (expected_width, expected_height))
                                
                        #         self.video_writer.write_frame(processed_frame)
                                
                        #     except Exception as e:
                        #         self.logger.error(f"Error writing processed frame: {e}")



                        if frame_number % 100 == 0 and video_doc:
                            video_doc.status = f"processing ({frame_number} frames done)"
                            video_doc.save()

        except KeyboardInterrupt:
            self.logger.info(f"KeyboardInterrupt received in Detector for {self.device_name}. Cleaning up...")
            status = "Terminated by user (KeyboardInterrupt)"
        except Exception as e:
            self.logger.error(f"Error in Detector: {e}")

            status = f"Error in Detector: {e}"
                # video_doc.save()
        finally:
            self.logger.info(f"Exiting process loop for {self.device_name} with status: {status}")
            if self.ip_camera.capture and self.ip_camera.capture.isOpened():
                self.logger.info(f"Releasing camera capture for {self.device_name}")
                self.ip_camera.capture.release()

            # Ensure writers are closed with the final status
            if self.save_video and hasattr(self, 'raw_video_writer') and self.raw_video_writer: # Assuming save_video refers to the processed video writer
                self.logger.info(f"Finishing processed video clip for {self.device_name}")
                # self.video_writer.finish_clip(status=status) # If you have a separate processed video writer
            
            if self.save_raw_video and self.raw_video_writer:
                self.logger.info(f"Finishing raw video clip for {self.device_name}")
                self.raw_video_writer.finish_clip(status=status)

            if hasattr(self, 'metadata_handler') and self.metadata_handler:
                self.logger.info(f"Closing metadata handler for {self.device_name}")
                self.metadata_handler.close()

            # if self.save_video and self.video_writer:
            #     self.video_writer.finish_clip(status="done")
            #     self.logger.info("Finished processed video writing")
                        
            # Disconnect database if it was connected by this instance
            # if self.database and self.database.is_connected(): # Add an is_connected method to your Database class
            #     self.logger.info(f"Disconnecting database for {self.device_name}")
            #     self.database.disconnect()

            self.logger.info(f"Cleaned up resources for {self.device_name}.")
            print(f"************************* Detector for {self.device_name} Completed/Terminated *************************\n")

    
    def process(self):
        """
        Main loop: read frames, detect humans, record metadata, optionally stream.
        """
        descriptor = f"Processing {self.device_name} Video"
        self.logger.info(descriptor)


        interval = datetime.timedelta(seconds=1 / self.target_fps)
        last_proc_time = None
        
        frame_number = 0
        initial_time_stamp = datetime.datetime.now(datetime.timezone.utc)
        video_doc = Cameras.objects(camera_address=self.video_path).first()
        self.no_frame_count = 0
        self.cam_result = True
        try:
        # if 1:
            # print("************************* Processing *************************\n")
            while self.ip_camera.is_open and self.cam_result:

                current_time = datetime.datetime.now(datetime.timezone.utc)

                # --- time‐based sampling logic ---
                if last_proc_time is not None and (current_time - last_proc_time) < interval:
                    # too soon: skip this frame
                    continue
                last_proc_time = current_time
                # ----------------------------------

                self.cam_result, frame = self.ip_camera.capture.read()

                if frame is None:
                    print("No frame captured, skipping...")
                    self.no_frame_count += 1
                    if self.no_frame_count >=10:
                        break
                    else:
                        continue

            
                
                frame_number = self.ip_camera.capture.get(cv2.CAP_PROP_POS_FRAMES)
                

                # if frame_number % self.ip_camera.process_skip_frame != 0:
                #     continue

                if frame is not None and self.cam_result:
                    
                    raw_frame = frame.copy()
                    
                    processed_frame = self.predict(raw_frame, current_time, frame_number)
                    # print(f"track_ids_dict: {track_ids_dict}")
                    if self.security_module:
                        self.security_module.process_frame(frame)
                    # if self.save_video:
                    #     self.video_writer.write_frame(processed_frame, current_time)



                    if frame_number % 100 == 0 and video_doc:
                        video_doc.status = f"processing ({frame_number} frames done)"
                        video_doc.save()


        except Exception as e:
            self.logger.error(f"Error in Detector: {e}")

            status = f"Error in Detector: {e}"
                # video_doc.save()

        finally:

            if self.ip_camera.capture:
                self.ip_camera.capture.release()  # Release the camera capture resource.
            # if self.database:
                # self.database.disconnect()  # Close database connection if open.


            status = "Finishin"
            if self.save_video:
                self.raw_video_writer.finish_clip(status=status)
            if self.save_raw_video:
                self.raw_video_writer.finish_clip(status=status)
# Ensure raw video writer is released properly.

            self.metadata_handler.close()
            self.ip_camera.capture.release()
            self.logger.info("Released VideoCapture object.")





            self.logger.info("Cleaned up all resources.")
            print("************************* Completed *************************\n")

