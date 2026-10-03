import cv2
from ultralytics import YOLO
import torch
import datetime
import numpy as np
from tqdm import tqdm
import os
import pprint
from shapely.geometry import Point, Polygon, LineString, box
from collections import deque

from src.utils import plot
from src.database.schemas.zone_schema import Zone
from src.database.database import Database
from src.database.metadata_handler import MetadataHandler
from src.database.schemas.cameras_schema import Videos

from src.core.actuators.video_writer_h265 import VideoWriterManager_h265
from src.core.actuators.streaming_machine import StreamingMachine


class HumanDetector:
    """
    HumanDetector machine that uses a YOLO model to detect humans in frames,
    track them, and handle zone/line-based logic.
    """

    def __init__(self, config, logger=None, save_video=False, save_raw_video=False):
        self.device = config.get('device', 'cuda:0')
        self.model_path = config.get('model_path')
        self.tracker_path = config.get('tracker_path')
        self.plot = config.get('plot', False)
        self.conf = config.get('conf', 0.6)
        self.class_ids = config.get('class_ids', [0,1])  # Default to 'person' class
        self.logger = logger

        # Initialize database connection
        self.database = Database()
        self.database.connect_db()

        # Sanity checks
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.tracker_path):
            raise FileNotFoundError(f"Tracker file not found at {self.tracker_path}")
        if not torch.cuda.is_available() and self.device.startswith('cuda'):
            self.logger.info("CUDA not available, switching to CPU.")
            self.device = 'cpu'

        # Load YOLO model
        try:
            self.logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path).to(self.device)
            self.logger.info("YOLO model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise
        self.model.tracker = self.tracker_path

        # IP Camera configs
        self.ip_camera = config.get('ip_camera')
        if not self.ip_camera:
            self.logger.error("No 'ip_camera' instance provided in config.")
            raise ValueError("IP Camera configuration is missing.")

        self.device_name = self.ip_camera.device_name
        self.is_video_file = self.ip_camera.is_video_file
        self.fps = self.ip_camera.fps

        # For zone-based checks
        self.zones = self.ip_camera.zones
        self.zones_iou = config.get("zones_iou", True)
        self.zone_dict = {}
        self.set_zones()
        self.conf_threshold = config.get('conf_threshold', 0.4)

        # Video path
        if self.is_video_file:
            self.video_path = self.ip_camera.ip_address  # Local video file path
        else:
            self.video_path = None  # Live feed

        # For storing track records
        # track_id -> {
        #   "records": deque of up to 25 detection snapshots,
        #   "start_time": datetime of first detection,
        #   "update_time": datetime of last detection,
        # }

        # Video writing if needed
        current_datetime = datetime.datetime.now(datetime.timezone.utc)()
        self.run_time_str = current_datetime.strftime("%d%m%Y%H%M%S")
        date = current_datetime.date()

        self.save_video = save_video
        self.save_raw_video = save_raw_video

        if self.save_video:
            video_dir = os.path.join(f"./results/videos/FM/{date}", self.device_name)
            self.video_writer = VideoWriterManager_h265(
                save_videos_flag=True,
                date=date,
                results_base_dir=video_dir,
                frame_size=(int(self.ip_camera.frame_width ), int(self.ip_camera.frame_height)),
                fps=self.fps,
                logger=self.logger
            )

        if self.save_raw_video:
            raw_video_dir = os.path.join(f"./results/raw_videos/FM/{date}", self.device_name)
            self.raw_video_writer = VideoWriterManager_h265(
                save_videos_flag=True,
                date=date,
                results_base_dir=raw_video_dir,
                frame_size=(int(self.ip_camera.frame_width), int(self.ip_camera.frame_height)),
                fps=self.fps,
                logger=self.logger
            )

        # MetadataHandler
        self.metadata_handler = MetadataHandler(self.ip_camera.ip_address, self.ip_camera.device_name, logger=self.logger)

        # StreamingMachine
        self.stream = True  # If you enable streaming
        if self.stream:
            self.stream_frame_size = (640, 480)
            self.stream_fps = 10

            self.rtsp_url = f"rtsp://localhost:8554/{self.device_name}"
            self.streaming_machine = StreamingMachine(
                device_name=self.device_name,
                frame_size=self.stream_frame_size,
                fps=self.stream_fps,
                rtsp_url=self.rtsp_url,
                logger=self.logger
            )

    def _get_center(self, bbox):
        """
        Calculate center of the bounding box [x1, y1, x2, y2].
        """
        x_center = (bbox[2])
        y_center = (bbox[1] + bbox[3]) / 2
        return (x_center, y_center)

    def set_zones(self):
        """
        Convert zone definitions to Shapely geometries and store in self.zone_dict,
        skipping any invalid shapes.
        """
        self.zone_dict = {}
        for zone_name, zone_id in self.zones.items():
            zone_doc = Zone.objects(id=zone_id).first()
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
                if poly.is_empty or not poly.is_valid:
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


    def _check_spatial_relationship(self, shape_info, point):
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

        # LINE
        elif shape_type == "line":
            line_coords = list(shape.coords)
            if len(line_coords) != 2:
                # Not a simple 2-point line; treat as on by default
                return "on"

            (x1, y1), (x2, y2) = line_coords
            px, py = point.x, point.y

            # Check orientation: is line "more horizontal" or "more vertical"?
            dx = x2 - x1
            dy = y2 - y1

            # If absolute horizontal span > vertical => line is "more horizontal"
            if abs(dx) > abs(dy):
                # Use y difference to decide above/below
                # We can param: find line eqn => y_line(px) and compare py
                # But let's do cross product approach and interpret sign for above/below
                # cross = (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)
                cross_val = dx * (py - y1) - dy * (px - x1)
                if abs(cross_val) < 1e-6:
                    return "on"
                elif cross_val > 0:
                    return "above"
                else:
                    return "below"
            else:
                # "more vertical", use x difference => left/right
                cross_val = dx * (py - y1) - dy * (px - x1)
                if abs(cross_val) < 1e-6:
                    return "on"
                elif cross_val > 0:
                    return "left"
                else:
                    return "right"

        return "unknown"

    def set_track_ids_status(self, track_ids_dict):
        """
        1) Store/update each track_id in self.track_ids_record with a 25-length deque of records.
        2) Determine where the bbox center is with respect to each zone or line.
        3) Compute movement/crossing by comparing the new state to the previous one.
        4) Remove old track_ids (>5 min without updates).
        """
        # current_time = datetime.datetime.now(datetime.timezone.utc)()

        for track_id, obj_info in track_ids_dict.items():

            # 2. Determine zone/line relationships
            center_pt = Point(self._get_center(obj_info["bbox"]))
            instance_dict = {}  # e.g. { zone_name: 'inside'/'outside'/'on', line_name: 'left'/'right'/'above'/'below'/'on' }
            for zone_name, zone_info in self.zone_dict.items():
                relation = self._check_spatial_relationship(zone_info, center_pt)
                instance_dict[zone_name] = relation

            obj_info["instance_dict"] = instance_dict


        return track_ids_dict

    def predict(self, frame, current_time):
        """
        Apply YOLO detection & tracking on the frame.
        Returns (track_ids_dict, frame).
        """
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device
        )



        track_ids_dict = {}
        if not results or len(results) == 0 or results[0].boxes.id is None:
            return track_ids_dict, frame

        # Extract detection data
        boxes = results[0].boxes.xyxy.int().cpu().numpy()      # [N,4]
        track_ids = results[0].boxes.id.int().cpu().numpy()    # [N]
        labels = results[0].boxes.cls.int().cpu().numpy()      # [N]
        confidence = results[0].boxes.conf.cpu().numpy()       # [N]
        class_names = results[0].names

        # Fill track_ids_dict
        for idx, track_id in enumerate(track_ids):
            # We'll embed run_time_str into the ID if desired, or keep track_id as is
            updated_track_id = int(f"{self.run_time_str}{track_id}")
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

        # If self.plot is True, draw the bounding boxes
        if self.plot:
            frame = results[0].plot(conf=False, labels=True, boxes=True, masks=False)
            # Optionally draw zone shapes for debugging
            if self.zone_dict:
                frame = plot.plot_shapes(frame, self.zone_dict)

        return track_ids_dict, frame

    def process(self):
        """
        Main loop: read frames, detect humans, record metadata, optionally stream.
        """
        descriptor = f"Processing {self.device_name} Video"
        self.logger.info(descriptor)

        if self.stream:
            self.streaming_machine.start_streaming()

        frame_number = 0
        initial_time_stamp = datetime.datetime.now(datetime.timezone.utc)()
        video_doc = Videos.objects(camera_address=self.video_path).first()

        self.cam_result = True
        try:
            while self.ip_camera.is_open and self.cam_result:
                if self.ip_camera.capture.read() is not None:
                    self.cam_result, frame = self.ip_camera.capture.read()
                frame_number = self.ip_camera.capture.get(cv2.CAP_PROP_POS_FRAMES)

                print("\n*****************************************************")
                print("device_name", self.device_name)
                print("cam_result", self.cam_result)
                print("frame.shape", frame.shape)
                print("frame_number", frame_number)




                raw_frame = frame
                # Skip frames if needed
                if frame_number % self.ip_camera.process_skip_frame != 0:
                    continue

                if frame is not None and self.cam_result:
                    track_ids_dict, processed_frame = self.predict(frame, current_time)

                    # Streaming
                    if self.stream:
                        # Stream the processed frame
                        self.streaming_machine.add_frame(cv2.resize(frame, self.stream_frame_size, interpolation=cv2.INTER_LINEAR))

                    # Handle video writing in 1-minute chunks
                    current_time = datetime.datetime.now(datetime.timezone.utc)()
                    if current_video_start_time is None:
                        # Start the first video file
                        current_video_start_time = current_time
                        current_video_path = self.raw_video_writer.start_new_video(current_video_start_time)
                        current_video_path = self.video_writer.start_new_video(current_video_start_time)


                    elapsed_time = (current_time - current_video_start_time).total_seconds()
                    if elapsed_time > 60:  # Switch to a new video file after 1 minute
                        self.video_writer.end_current_video()
                        self.raw_video_writer.end_current_video()
                        current_video_start_time = current_time
                        current_video_path = self.raw_video_writer.start_new_video(current_video_start_time)
                        current_video_path = self.video_writer.start_new_video(current_video_start_time)


                    self.video_writer.write_frame(frame)
                    self.raw_video_writer.write_frame(raw_frame)

                    if track_ids_dict is not None:
                        
                        # Optional: zone checks

                        updated_track_id = []
                        for track_id in track_ids_dict:
                            updated_track_id.append(f"{self.run_time_str}_{track_id}")

                        if self.zones_iou:
                            zones_track_id = self.zones_iou_check(track_ids_dict["labels"], track_ids_dict["boxes"], updated_track_id, track_ids_dictconfidence)
                        else:
                            zones_track_id = {}

                        processed_data = {
                            "time_stamp": current_time,
                            "current_video_path": self.video_writer.current_video_path,

                            "frame_number": frame_number,
                            "track_ids_info": track_ids_dict
                            # Add other info as needed...
                        }


                        self.metadata_handler.process(processed_data)

                        if frame_number % 100 == 0 and video_doc:
                            video_doc.status = f"processing ({frame_number} frames done)"
                            video_doc.save()
                            print("\nFrame Number: ", frame_number)
                            pprint.pprint(processed_data)

        except Exception as e:
            self.logger.error(f"Error in HumanDetector: {e}")
            if video_doc:
                video_doc.status = f"Error in HumanDetector: {e}"
                video_doc.save()

        finally:
            if video_doc:
                end_time = datetime.datetime.now(datetime.timezone.utc)()
                time_diff = end_time - initial_time_stamp
                hours = time_diff.seconds // 3600
                minutes = (time_diff.seconds % 3600) // 60
                seconds = time_diff.seconds % 60
                time_str = f"{hours}h {minutes}m {seconds}s"
                video_doc.status = f"Completed {int(frame_number)} in {time_str}"
                video_doc.save()

            self.metadata_handler.close()
            self.ip_camera.capture.release()
            self.logger.info("Released VideoCapture object.")

            if self.save_video:
                self.video_writer.end_current_video()
            if self.save_raw_video:
                self.raw_video_writer.end_current_video()

            cv2.destroyAllWindows()

            if self.stream:
                self.streaming_machine.stop_streaming()

            self.logger.info("Cleaned up all resources.")
            print("************************* Completed *************************\n")
