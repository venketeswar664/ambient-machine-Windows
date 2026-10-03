# rt_detr_detector.py

import os
import cv2
import logging
import queue
import threading
import datetime
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from shapely.geometry import Point, Polygon, LineString, box

import supervision as sv
from src.utils import plot
from src.database.schemas.sentinel_poc_schema import Cameras, CameraZoneMapping
from src.database.metadata_handler import MetadataHandler
from src.database.database import Database
from src.core.actuators.video_writer_manager import VideoWriterManager
from src.core.think.machine_learning.computer_vision.safety_tracker import SecurityModule

from src.core.think.machine_learning.computer_vision.trt_inference import TRTInference


# COCO class map (0..2)
COCO_CLASSES = {
    0: 'employee',
    1: 'customer',
    2: 'background'
}


class Detector:
    """
    RT-DETR + ByteTrack–based detector that:
      1. Loads a TensorRT engine via TRTInference
      2. Preprocesses each incoming frame (already rotated) → Tensor
      3. Runs inference → obtains raw boxes/scores/labels
      4. Filters by confidence & class IDs → rescales boxes to original frame size
      5. Wraps into supervision.Detections → tracks via ByteTrack
      6. Computes zone IoU/location → optionally stores crops
      7. Writes metadata via MetadataHandler
      8. Optionally annotates & saves video frames
      9. Properly cleans up TRT resources on close
    """

    def __init__(self,
                 config: dict,
                 logger = None,
                 shutdown_event: threading.Event = None):
        """
        :param config: dict containing:
            - "device": e.g., "cuda:0" or "cpu"
            - "trt_engine_path": path to .engine file
            - "trt_input_size": (H, W), typically (640, 640)
            - "conf": float confidence threshold (e.g. 0.5)
            - "class_ids": list of int COCO class IDs to keep (e.g. [0,1])
            - "plot": bool (True to draw boxes + zones on output)
            - "save_raw_video": bool
            - "raw_video_fps": int
            - "save_processed_video": bool
            - "processed_video_fps": int
            - "store_crops": bool
            - "security_enabled": bool
            - "ip_camera": an object that has:
                 • device_name (str)
                 • is_video_file (bool)
                 • fps (int/float)
                 • ip_address (str)
                 • frame_width, frame_height (int)
                 • process_skip_frame (int)
                 • zones (list of Zone references)
                 • store_name (str)
                 • Methods: .connect(), .close(), .is_open, .capture.read()
            - "zones_iou": bool (True to compute IoU vs. zones)
        :param logger: instance of your custom Logger (wraps a Python logger)
        :param shutdown_event: threading.Event to signal immediate shutdown
        """
        # Logger: if a custom Logger wrapper was passed in, unwrap it; otherwise assume it's a Python logger.
        if hasattr(logger, "logger"):
            self.logger = logger.logger
        else:
            self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Ensure at least one handler exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing Detector...")

        # --- Basic config parameters ---
        self.device = config.get("device", "cuda:0")
        self.trt_engine_path = config.get("trt_engine_path")
        self.trt_input_size = config.get("trt_input_size", (640, 640))
        self.conf = config.get("conf", 0.5)
        self.class_ids = config.get("class_ids", [0])  # e.g. [0, 1, 2]
        self.plot = config.get("plot", False)

        self.store_crops_flag = config.get("store_crops", True)
        self.zones_iou = config.get("zones_iou", True)

        self.shutdown_event = shutdown_event

        # --- Connect to database ---
        self.database = Database()
        self.database.connect_db()

        # --- Validate TRT engine path ---
        if not self.trt_engine_path or not os.path.isfile(self.trt_engine_path):
            self.logger.error(f"TensorRT engine file not found: {self.trt_engine_path}")
            raise FileNotFoundError(f"TensorRT engine file not found: {self.trt_engine_path}")

        # --- Check CUDA availability if using GPU ---
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("CUDA not available; switching to CPU.")
            self.device = "cpu"

        # --- Load TRT model ---
        self.trt_model = None
        try:
            self.logger.info(f"Loading TRTInference model from: {self.trt_engine_path}")
            self.trt_model = TRTInference(
                engine_path=self.trt_engine_path,
                device=self.device,
                backend="torch",
                max_batch_size=1,
                verbose=False,
                logger=self.logger.getChild("TRTInference")
            )
            self.logger.info("RT-DETR TRT model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load TRT model: {e}")
            if self.trt_model is not None:
                try:
                    self.trt_model.destroy()
                except Exception:
                    pass
                self.trt_model = None
            raise

        # --- Build a mapping from logical names → actual TRT output tensor names ---
        self.model_output_names = {
            "boxes":   next((n for n in self.trt_model.output_names if "box" in n.lower()),   None),
            "scores":  next((n for n in self.trt_model.output_names if "score" in n.lower()), None),
            "labels":  next((n for n in self.trt_model.output_names if "label" in n.lower()), None),
            "num_dets": next((n for n in self.trt_model.output_names if "num_det" in n.lower()), None)
        }
        self.logger.debug(f"Model output name mapping: {self.model_output_names}")

        # --- Initialize ByteTrack tracker & annotators ---
        # Note: ByteTrack takes no constructor arguments by default
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.7,
            text_padding=5,
            text_position=sv.Position.TOP_LEFT
        )
        self.class_names_map = COCO_CLASSES

        # --- IP Camera setup ---
        self.ip_camera = config.get("ip_camera")
        if self.ip_camera is None:
            self.logger.error("No 'ip_camera' provided in config.")
            raise ValueError("IP Camera configuration is missing.")

        self.device_name = self.ip_camera.device_name
        self.is_video_file = self.ip_camera.is_video_file
        self.fps = self.ip_camera.fps
        self.process_skip_frame = self.ip_camera.process_skip_frame
        self.store_name = getattr(self.ip_camera, "store_name", "default_store")

        # --- Security module (initialized once a frame is available) ---
        if config.get("security_enabled", False):
            self.config_security_enabled = True
            self.security_module = None
        else:
            self.config_security_enabled = False
            self.security_module = None

        # --- Zones setup ---
        self.zones = self.ip_camera.zones
        self.zone_dict = {}
        self.set_zones()

        self.rotation = self.ip_camera.rotation

        # --- Video writer setup ---
        # ip_camera now gives already-rotated frames, so use those dims directly:
        frame_w = int(self.ip_camera.frame_width)
        frame_h = int(self.ip_camera.frame_height)


        self.target_fps = 5
        self.frame_interval = datetime.timedelta(seconds=1 / self.target_fps)
        self._last_process_time = None


        self.save_raw_video = config.get("save_raw_video", True)
        self.raw_video_fps = config.get("raw_video_fps", 5)

        self.save_processed_video = config.get("save_processed_video", False)
        self.processed_video_fps = config.get("processed_video_fps", 5)

        # Raw (un-annotated) video writer
        self.raw_video_writer = None
        if self.save_raw_video:
            date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            raw_dir = os.path.join("results", "raw_videos", self.store_name, date_str, self.device_name)
            os.makedirs(raw_dir, exist_ok=True)
            self.raw_video_writer = VideoWriterManager(
                camera_id=self.device_name,
                output_dir_template=raw_dir,
                frame_width=frame_w,
                frame_height=frame_h,
                target_fps=self.raw_video_fps,
                source_fps_val=self.fps,
                original_stream_width=frame_w,
                original_stream_height=frame_h,
                parent_logger=self.logger.getChild("RawVideoWriter")
            )

        # Processed (annotated) video writer
        self.processed_video_writer = None
        if self.save_processed_video:
            date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            proc_dir = os.path.join("results", "processed_videos", self.store_name, date_str, self.device_name)
            os.makedirs(proc_dir, exist_ok=True)
            self.processed_video_writer = VideoWriterManager(
                camera_id=self.device_name + "_processed",
                output_dir_template=proc_dir,
                frame_width=frame_w,
                frame_height=frame_h,
                target_fps=self.processed_video_fps,
                source_fps_val=self.fps,
                original_stream_width=frame_w,
                original_stream_height=frame_h,
                parent_logger=self.logger.getChild("ProcessedVideoWriter")
            )

        # --- Crop saving setup ---
        self.crop_dir = None
        if self.store_crops_flag:
            self.crop_dir = os.path.join("results", "track_ids", self.store_name)

        # --- Metadata handler (MongoDB) ---
        self.metadata_handler = MetadataHandler(
            self.ip_camera.ip_address,
            self.device_name,
            logger=self.logger.getChild("MetadataHandler")
        )

        # --- Run ID string for unique track IDs ---
        self.run_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%d%m%Y%H%M%S%f")

        # --- Pre-compiled transform for TRT input ---
        self.transforms_for_trt = T.Compose([
            T.Resize(self.trt_input_size),
            T.ToTensor()
        ])

        # --- Keep track of original frame size (updated each inference) ---
        self.h_orig = frame_h
        self.w_orig = frame_w

        # --- State flags ---
        self._is_closed = False
        self.logger.info("Detector initialization complete.")


    # -------------------------------------------------------------------------
    # ZONE LOADING
    # -------------------------------------------------------------------------
    def set_zones(self):
        """
        Populate self.zone_dict = {
            zone_name: {"shape": Polygon or LineString, "type": "zone" or "line"}
        } for the camera referenced by self.device_name.
        """
        self.zone_dict = {}
        camera_device_name = self.device_name
        if not camera_device_name:
            self.logger.error("Camera device_name is missing in set_zones().")
            return

        camera_doc = Cameras.objects(device_name=camera_device_name).first()
        if not camera_doc:
            self.logger.error(f"Camera '{camera_device_name}' not found in DB.")
            return

        mappings = CameraZoneMapping.objects(camera=camera_doc)
        for mapping in mappings:
            zone = mapping.zone
            zone_name = zone.name
            roi = mapping.roi  # Expect a list of coordinates, e.g. [[x1,y1,x2,y2,...]]
            if not roi or not isinstance(roi, list) or len(roi) < 1:
                self.logger.warning(f"No ROI found for zone '{zone_name}'. Skipping.")
                continue

            coords = roi[0]
            if not isinstance(coords, list) or len(coords) % 2 != 0:
                self.logger.warning(f"Invalid ROI format for zone '{zone_name}'. Skipping.")
                continue

            pts = [tuple(coords[i:i + 2]) for i in range(0, len(coords), 2)]
            if len(pts) > 2:
                poly = Polygon(pts)
                if not poly.is_valid or poly.is_empty:
                    self.logger.warning(f"Invalid polygon for zone '{zone_name}'. Skipping.")
                    continue
                self.zone_dict[zone_name] = {"shape": poly, "type": "zone"}
            elif len(pts) == 2:
                line = LineString(pts)
                if not line.is_valid or line.is_empty:
                    self.logger.warning(f"Invalid line for zone '{zone_name}'. Skipping.")
                    continue
                self.zone_dict[zone_name] = {"shape": line, "type": "line"}
            else:
                self.logger.warning(f"Insufficient points for zone '{zone_name}'. Skipping.")
                continue

        self.logger.info(f"Loaded {len(self.zone_dict)} zones for camera '{camera_device_name}'.")


    # -------------------------------------------------------------------------
    # I-O/U + Location Helpers
    # -------------------------------------------------------------------------
    def _get_center(self, bbox_coords):
        """
        Compute (x_center, y_bottom) for a bbox [x1,y1,x2,y2].
        """
        x1, y1, x2, y2 = bbox_coords[:4]
        return ((x1 + x2) / 2.0, y2)

    def _calculate_iou(self, bbox_coords, zone_polygon):
        """
        Given bbox_coords = [x1,y1,x2,y2], and a shapely Polygon, return IoU.
        """
        try:
            x1, y1, x2, y2 = bbox_coords[:4]
            bbox_poly = box(x1, y1, x2, y2)
            if not zone_polygon.is_valid or zone_polygon.is_empty:
                return 0.0
            inter_area = bbox_poly.intersection(zone_polygon).area
            union_area = bbox_poly.area
            return round(inter_area / union_area, 4) if union_area > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Error computing IoU: {e}")
            return 0.0

    def _check_spatial_relationship_iou(self, shape_info, bbox_coords):
        """
        If shape_info['type']=="zone", compute IoU vs. that polygon and return it.
        bbox_coords should be [x1,y1,x2,y2].
        """
        if not isinstance(bbox_coords, (list, tuple, np.ndarray)) or len(bbox_coords) < 4:
            self.logger.warning(f"Invalid bbox_coords for IoU: {bbox_coords}")
            return 0.0

        shape = shape_info["shape"]
        shape_type = shape_info["type"]
        if shape_type == "zone" and isinstance(shape, Polygon) and shape.is_valid:
            return self._calculate_iou(bbox_coords, shape)
        return 0.0

    def _check_spatial_relationship_location(self, shape_info, center_coords):
        """
        For polygons: return "inside", "on", or "outside".
        For lines: return "above"/"below"/"on"/"left"/"right".
        """
        shape = shape_info["shape"]
        shape_type = shape_info["type"]
        pt = Point(center_coords)

        if shape_type == "zone":
            if shape.contains(pt):
                return "inside"
            if shape.touches(pt):
                return "on"
            return "outside"

        if shape_type == "line":
            coords = list(shape.coords)
            if len(coords) != 2:
                return "on"
            (x1, y1), (x2, y2) = coords
            dx, dy = x2 - x1, y2 - y1
            cross = dx * (pt.y - y1) - dy * (pt.x - x1)
            eps = 1e-6
            if abs(cross) < eps:
                return "on"
            if abs(dx) > abs(dy):
                return "above" if cross > 0 else "below"
            else:
                return "left" if cross > 0 else "right"

        return "unknown"


    # -------------------------------------------------------------------------
    # TRACK & ZONE STATUS Assignment
    # -------------------------------------------------------------------------
    def set_track_ids_status(self, track_ids_dict: dict) -> dict:
        """
        For each track in track_ids_dict:
          - Compute IoU vs. every zone polygon
          - Compute location (inside/on/outside or above/below/on/left/right)
          - Store results in obj_info['instance_dict'][zone_name] = {"iou":..., "location":...}
        """
        for track_id, obj_info in track_ids_dict.items():
            bbox_coords = obj_info["bbox"]
            instance_dict = {}

            for zone_name, shape_info in self.zone_dict.items():
                iou_v = self._check_spatial_relationship_iou(shape_info, bbox_coords)
                center = self._get_center(bbox_coords)
                loc_stat = self._check_spatial_relationship_location(shape_info, center)
                instance_dict.setdefault(zone_name, {})["iou"] = iou_v
                instance_dict[zone_name]["location"] = loc_stat

            obj_info["instance_dict"] = instance_dict

        return track_ids_dict


    # -------------------------------------------------------------------------
    # RT-DETR Inference Helper (with orig_target_sizes support)
    # -------------------------------------------------------------------------
    def _run_rt_detr_and_track(self, frame_bgr: np.ndarray):
        """
        Given a BGR frame (already rotated),
        resize → tensor → TRTInference → return outputs dict.
        Tracking is performed later in process_detection_results().
        """
        if self.trt_model is None:
            self.logger.error("TRT model not initialized.")
            return {}

        # Save original frame dimensions
        self.h_orig, self.w_orig = frame_bgr.shape[:2]

        # 1) BGR→RGB→PIL→Resize→Tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        try:
            resized_tensor = self.transforms_for_trt(pil_img).unsqueeze(0).to(self.trt_model.device)
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            return {}

        # 2) Build blob dict.
        blob = {}
        input_names = self.trt_model.input_names  # e.g. ["images", "orig_target_sizes"] or just ["images"]
        blob[input_names[0]] = resized_tensor

        if len(input_names) > 1:
            orig_size_tensor = torch.tensor([[self.w_orig, self.h_orig]],
                                            dtype=torch.long).to(self.trt_model.device)
            blob[input_names[1]] = orig_size_tensor

        # 3) Inference
        try:
            trt_outputs = self.trt_model(blob)
        except Exception as e:
            self.logger.error(f"Error during TRT inference: {e}")
            return {}

        return trt_outputs


    # -------------------------------------------------------------------------
    # PUBLIC PREDICT: inference + spawn background thread
    # -------------------------------------------------------------------------
    def predict(self, frame: np.ndarray, current_time: datetime.datetime, frame_number: int):
        """
        1) Run inference on the frame to get raw outputs
        2) Spawn a background thread that will:
           - Filter outputs → ByteTrack track
           - Compute IoU + location → save crops → metadata
           - Annotate + save processed frame
           - Save raw video frame → security processing
        """
        # 1) Inference (no rotation, camera gives already-rotated frames)
        trt_outputs = self._run_rt_detr_and_track(frame)

        # 2) Dispatch to background thread for post-processing
        _ = self.process_detection_results_thread(
            trt_outputs,
            frame.copy(),
            current_time,
            frame_number
        )

        return frame_number


    # -------------------------------------------------------------------------
    # BACKGROUND THREAD: post-process TRT outputs
    # -------------------------------------------------------------------------
    def process_detection_results_thread(self,
                                         trt_outputs: dict,
                                         frame_proc: np.ndarray,
                                         current_time: datetime.datetime,
                                         frame_number: int) -> threading.Thread:
        """
        Spawn a daemon thread that will:
          - Read TRT outputs → boxes/scores/labels
          - Filter → rescale → ByteTrack track
          - Compute zones IoU/location
          - Save crops
          - Write metadata via MetadataHandler
          - Annotate + save processed video frame
          - Save raw video frame → security processing
        """
        arg_q = queue.Queue(maxsize=1)
        arg_q.put((trt_outputs, frame_proc, current_time, frame_number))
        thread = threading.Thread(target=self.process_detection_results, args=(arg_q,))
        thread.daemon = True
        thread.start()
        return thread

    def process_detection_results(self, arg_q: queue.Queue):
        """
        Worker method for the background thread.
        Steps:
          1) Map TRT outputs → boxes_raw/scores_raw/labels_raw
          2) Filter by num_dets/conf/class → rescale to (w_orig, h_orig)
          3) Wrap into supervision.Detections → ByteTrack.update_with_detections(...)
          4) Save raw frame if enabled
          5) Build track_ids_dict → zones IoU/location → store crops → metadata
          6) Annotate (boxes+IDs+zones) → save processed frame
          7) Security processing
        """
        try:
            trt_outputs, frame_proc, ts, fnum = arg_q.get()
            arg_q.task_done()

            # --- Early exit if inference dict is empty or missing required keys ---
            if not trt_outputs:
                return

            # -----------------------------------------------------------------
            # STEP 1: Grab “boxes”, “scores”, “labels” from TRT outputs
            # -----------------------------------------------------------------
            boxes_key  = self.model_output_names.get("boxes")
            scores_key = self.model_output_names.get("scores")
            labels_key = self.model_output_names.get("labels")

            raw_boxes_tensor  = trt_outputs.get(boxes_key)
            raw_scores_tensor = trt_outputs.get(scores_key)
            raw_labels_tensor = trt_outputs.get(labels_key)

            if raw_boxes_tensor is None or raw_scores_tensor is None or raw_labels_tensor is None:
                self.logger.error(
                    f"[{self.device_name}] Critical output tensors missing from trt_outputs. "
                    f"Boxes: {raw_boxes_tensor is not None}, "
                    f"Scores: {raw_scores_tensor is not None}, "
                    f"Labels: {raw_labels_tensor is not None}. Skipping frame."
                )
                return

            try:
                # Batch size = 1, so index [0]
                boxes_raw  = raw_boxes_tensor[0].cpu().numpy()
                scores_raw = raw_scores_tensor[0].cpu().numpy()
                labels_raw = raw_labels_tensor[0].cpu().numpy()
            except (AttributeError, IndexError, KeyError, TypeError) as e:
                self.logger.error(
                    f"[{self.device_name}] Error accessing or processing TRT outputs. "
                    f"Config: {self.model_output_names}. Data type: {type(trt_outputs)}. "
                    f"Error: {e}", exc_info=True
                )
                return

            # Determine number of detections if available
            num_dets = -1
            num_dets_key = self.model_output_names.get("num_dets")
            if num_dets_key and num_dets_key in trt_outputs:
                try:
                    num_dets_tensor = trt_outputs[num_dets_key]
                    # If it's a length-1 tensor
                    if num_dets_tensor is not None and hasattr(num_dets_tensor, '__len__') and len(num_dets_tensor) > 0 and hasattr(num_dets_tensor[0], 'item'):
                        num_dets = int(num_dets_tensor[0].item())
                    elif torch.is_tensor(num_dets_tensor) and num_dets_tensor.numel() == 1:
                        num_dets = int(num_dets_tensor.item())
                except Exception as e_num_dets:
                    self.logger.warning(
                        f"[{self.device_name}] Could not parse num_dets from '{num_dets_key}': {e_num_dets}. "
                        f"Using all available detections."
                    )

            # Slice by num_dets if valid, else keep all
            if 0 <= num_dets <= boxes_raw.shape[0]:
                boxes, scores, labels = (
                    boxes_raw[:num_dets],
                    scores_raw[:num_dets],
                    labels_raw[:num_dets]
                )
            else:
                boxes, scores, labels = boxes_raw, scores_raw, labels_raw

            # -----------------------------------------------------------------
            # STEP 2: Filter by confidence threshold and allowed class IDs
            # -----------------------------------------------------------------
            mask_conf = (scores >= self.conf) & (np.isin(labels, self.class_ids))
            boxes_filtered  = boxes[mask_conf]
            scores_filtered = scores[mask_conf]
            labels_filtered = labels[mask_conf].astype(int)

            # If nothing remains, bail out
            if boxes_filtered.shape[0] == 0:
                return

            # -----------------------------------------------------------------
            # STEP 3: Create supervision.Detections and run ByteTrack
            # -----------------------------------------------------------------
            try:
                detections_sv = sv.Detections(
                    xyxy=boxes_filtered,
                    confidence=scores_filtered,
                    class_id=labels_filtered
                )
            except Exception as e_sv:
                self.logger.error(f"[{self.device_name}] Error creating sv.Detections: {e_sv}", exc_info=True)
                detections_sv = sv.Detections.empty()

            # ByteTrack tracking (fills detections_sv.tracker_id, etc.)
            try:
                tracked_detections = self.tracker.update_with_detections(detections_sv)
            except Exception as e_tracker:
                self.logger.error(
                    f"[{self.device_name}] Error updating ByteTrack: {e_tracker}. "
                    f"Using untracked detections_sv for this frame.",
                    exc_info=True
                )
                tracked_detections = detections_sv  # fallback to no tracking

            # If even tracked_detections is empty or has no boxes, bail
            if tracked_detections is None or len(tracked_detections.xyxy) == 0:
                return

            # -----------------------------------------------------------------
            # STEP 4: Save raw frame (if requested)
            # -----------------------------------------------------------------
            if self.save_raw_video and self.raw_video_writer:
                self.raw_video_writer.write_frame(frame_proc)

            # -----------------------------------------------------------------
            # STEP 5: Build track_ids_dict (with unique IDs, class names, etc.)
            # -----------------------------------------------------------------
            track_ids_dict = {}
            frame_pos = fnum  # our “frame position” is simply fnum
            for i in range(len(tracked_detections.xyxy)):
                # If i >= length of tracker_id array, warn + break
                if tracked_detections.tracker_id is None or i >= len(tracked_detections.tracker_id):
                    self.logger.warning(
                        f"[{self.device_name}] Mismatch in tracked_detections lengths. "
                        f"Skipping index {i}."
                    )
                    break

                track_id_val = tracked_detections.tracker_id[i]
                if track_id_val is None:
                    continue  # skip if no ID

                try:
                    unique_track_id = f"{self.run_time_str}_{int(track_id_val):04d}"
                except ValueError:
                    self.logger.warning(
                        f"[{self.device_name}] Could not convert track_id '{track_id_val}' to int. Skipping."
                    )
                    continue

                # Retrieve class_id and confidence safely
                class_id_val = (
                    int(tracked_detections.class_id[i])
                    if (tracked_detections.class_id is not None and i < len(tracked_detections.class_id))
                    else -1
                )
                confidence_val = (
                    float(tracked_detections.confidence[i])
                    if (tracked_detections.confidence is not None and i < len(tracked_detections.confidence))
                    else 0.0
                )

                # Build the dictionary entry
                track_ids_dict[unique_track_id] = {
                    "track_id":     unique_track_id,
                    "bbox":         tracked_detections.xyxy[i].astype(int).tolist(),
                    "label":        class_id_val,
                    "confidence":   confidence_val,
                    "label_name":   self.class_names_map.get(class_id_val, f'class_{class_id_val}'),
                    "current_time": ts,
                    "frame_pos":    int(frame_pos)
                }

            # -----------------------------------------------------------------
            # STEP 6: Compute zones IoU + location (if requested)
            # -----------------------------------------------------------------
            if self.zones_iou and track_ids_dict:
                track_ids_dict = self.set_track_ids_status(track_ids_dict)

            # -----------------------------------------------------------------
            # STEP 7: Save crops (if requested)
            # -----------------------------------------------------------------
            if self.store_crops_flag and track_ids_dict:
                track_ids_dict = self.store_crops(frame_proc, track_ids_dict)

            # -----------------------------------------------------------------
            # STEP 8: Build processed_data & write metadata
            # -----------------------------------------------------------------
            processed_data = {
                "time_stamp": ts,
                "evidence_path": (
                    self.raw_video_writer.get_output_path() if self.raw_video_writer else None
                ),
                "frame_number": fnum,
                "evidence_frame_number": (
                    self.raw_video_writer.frame_number if self.raw_video_writer else -1
                ),
                "track_ids_info": track_ids_dict
            }
            try:
                self.metadata_handler.process(processed_data)
            except Exception as e_meta:
                self.logger.error(f"MetadataHandler.process(...) failed: {e_meta}", exc_info=True)

            # -----------------------------------------------------------------
            # STEP 9: Security processing (if enabled)
            # -----------------------------------------------------------------
            if self.security_module:
                try:
                    self.security_module.process_frame(frame_proc)
                except Exception as e_sec:
                    self.logger.error(f"SecurityModule.process_frame(...) failed: {e_sec}", exc_info=True)

            # -----------------------------------------------------------------
            # STEP 10: Optionally annotate (boxes + zone outlines) & save processed video
            # -----------------------------------------------------------------
            if self.plot or self.save_processed_video:
                output_frame = frame_proc.copy()

                # 10.1) Draw bounding boxes + IDs
                if self.plot and len(tracked_detections.xyxy) > 0:
                    labels_list = []
                    for i in range(len(tracked_detections.xyxy)):
                        tid = int(tracked_detections.tracker_id[i])
                        cid = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else -1
                        conf = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 0.0
                        class_name = self.class_names_map.get(cid, f'class_{cid}')
                        labels_list.append(f"ID:{tid} {class_name} {conf:.2f}")

                    try:
                        output_frame = self.box_annotator.annotate(scene=output_frame, detections=tracked_detections)
                        if labels_list:
                            output_frame = self.label_annotator.annotate(
                                scene=output_frame,
                                detections=tracked_detections,
                                labels=labels_list
                            )
                    except Exception as e_annot:
                        self.logger.warning(f"Annotation failed: {e_annot}")

                # 10.2) Overlay zone polygons (if requested)
                if self.plot and self.zone_dict:
                    try:
                        output_frame = plot.plot_shapes(output_frame, self.zone_dict)
                    except Exception as e_plot:
                        self.logger.warning(f"Zone overlay failed: {e_plot}")

                # 10.3) Save annotated frame to processed video, if enabled
                if self.save_processed_video and self.processed_video_writer:
                    self.processed_video_writer.write_frame(output_frame)

        except Exception as e:
            self.logger.error(f"Exception in process_detection_results: {e}", exc_info=True)



    # -------------------------------------------------------------------------
    # CROP SAVING (unchanged logic)
    # -------------------------------------------------------------------------
    def store_crops(self, frame: np.ndarray, track_ids_dict: dict) -> dict:
        """
        For each track in track_ids_dict, crop the frame + padding, then:
          - If 'inside' any zone, save there
          - Else save under "no-zone"
        Append saved paths into obj_info["track_id_path_list"].
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        date_str = now_utc.date().isoformat()

        for track_id, obj_info in track_ids_dict.items():
            bbox = obj_info["bbox"]
            label = obj_info["label_name"]
            ts_for_fn = obj_info["current_time"].strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]

            x1, y1, x2, y2 = map(int, bbox)
            P = 10
            h, w = frame.shape[:2]
            x1p, y1p = max(0, x1 - P), max(0, y1 - P)
            x2p, y2p = min(w, x2 + P), min(h, y2 + P)

            if x2p <= x1p or y2p <= y1p:
                continue

            crop_img = frame[y1p:y2p, x1p:x2p]
            if crop_img.size == 0:
                continue

            instance_info = obj_info.get("instance_dict", {})
            obj_info.setdefault("track_id_path_list", [])

            saved_flag = False
            # If inside any zone, save there and break
            for zone_name, zone_data in instance_info.items():
                if zone_data.get("location") == "inside":
                    zm = zone_name.replace(" ", "_")
                    fname = f"{ts_for_fn}-{label}-inside-{zm}.jpg"
                    dir_path = os.path.join(self.crop_dir, date_str, self.device_name, zm, track_id)
                    os.makedirs(dir_path, exist_ok=True)
                    out_fp = os.path.join(dir_path, fname)
                    if cv2.imwrite(out_fp, crop_img):
                        obj_info["track_id_path_list"].append(out_fp)
                        saved_flag = True
                    break

            # If not saved in any zone, save under "no-zone"
            if not saved_flag:
                no_zone = "no-zone"
                fname = f"{ts_for_fn}-{label}-_-{no_zone}.jpg"
                dir_path = os.path.join(self.crop_dir, date_str, self.device_name, no_zone, track_id)
                os.makedirs(dir_path, exist_ok=True)
                out_fp = os.path.join(dir_path, fname)
                if cv2.imwrite(out_fp, crop_img):
                    obj_info["track_id_path_list"].append(out_fp)

        return track_ids_dict


    # -------------------------------------------------------------------------
    # SEGMENT FILENAME HELPER (unchanged)
    # -------------------------------------------------------------------------
    def segment_for(self, ts: datetime.datetime) -> str:
        seg_start = ts.replace(second=0, microsecond=0)
        date_folder = seg_start.strftime("%Y-%m-%d")
        fname = f"video_{seg_start.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        return os.path.join("results", "videos", self.store_name, date_folder, self.device_name, fname)


    # -------------------------------------------------------------------------
    # SECURITY MODULE INITIALIZATION
    # -------------------------------------------------------------------------
    def _initialize_security_module_if_needed(self, first_frame: np.ndarray):
        """
        If security is enabled and module not yet created, attempt to initialize it using first_frame.
        """
        if self.config_security_enabled and self.security_module is None and first_frame is not None:
            self.logger.info("Initializing Security Module with the first valid frame...")
            try:
                self.security_module = SecurityModule(self.device_name, first_frame.copy())
                self.logger.info("Security Module initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Security Module: {e}")

    # -------------------------------------------------------------------------
    # MAIN LOOP FOR VIDEO FILE MODE
    # -------------------------------------------------------------------------
    def process_video(self):
        """
        Process a local video file (self.ip_camera.ip_address) frame by frame:
         - Read, skip frames, predict, track, metadata, etc.
        """
        self.logger.info(f"Starting process_video for camera '{self.device_name}' (file mode).")

        if not self.is_video_file or not self.ip_camera.ip_address or not os.path.isfile(self.ip_camera.ip_address):
            self.logger.error(f"Video file path invalid: {self.ip_camera.ip_address}")
            self.close()
            return

        status = "Processing"
        try:
            # 1) Connect IP camera (in file mode, this should open cv2.VideoCapture(path))
            self.ip_camera.connect()
            if not self.ip_camera.is_open:
                self.logger.error(f"Failed to open video file: {self.ip_camera.ip_address}")
                status = "OpenFailed"
                self.close()
                return

            first_frame_for_security = None

            while self.ip_camera.is_open:
                if self.shutdown_event and self.shutdown_event.is_set():
                    self.logger.info("Shutdown event detected. Exiting process_video loop.")
                    status = "Shutdown"
                    break

                ret, frame = self.ip_camera.capture.read()
                if not ret or frame is None:
                    self.logger.info("End of video file or read error. Exiting.")
                    status = "EOF"
                    break

                # Initialize SecurityModule on first valid frame
                if first_frame_for_security is None:
                    self._initialize_security_module_if_needed(frame)
                    first_frame_for_security = frame

                current_time = datetime.datetime.now(datetime.timezone.utc)
                frame_number = int(self.ip_camera.capture.get(cv2.CAP_PROP_POS_FRAMES))

                # Run prediction (inference + tracking + background metadata)
                _ = self.predict(frame, current_time, frame_number)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt caught in process_video.")
            status = "KeyboardInterrupt"
        except Exception as e:
            self.logger.error(f"Error in process_video: {e}", exc_info=True)
            status = f"Error:{e}"
        finally:
            self.logger.info(f"Cleaning up after process_video with status: {status}")
            if self.ip_camera and self.ip_camera.is_open:
                self.ip_camera.close()
            if self.raw_video_writer:
                self.raw_video_writer.finish_clip(status=status)
            if self.processed_video_writer:
                self.processed_video_writer.finish_clip(status=status)
            if self.metadata_handler:
                self.metadata_handler.close()
            self.close()
            self.logger.info(f"Resources cleaned up for camera '{self.device_name}' (file mode).")


    # -------------------------------------------------------------------------
    # MAIN LOOP FOR LIVE/RTSP MODE
    # -------------------------------------------------------------------------

    def process(self):
        """
        Process a live RTSP stream (self.ip_camera.ip_address) frame by frame,
        but only run inference/tracking at ~self.target_fps.
        """
        self.logger.info(f"Starting process for camera '{self.device_name}' (live mode).")

        status = "Processing"
        try:
            # 1) Connect to IP camera
            self.ip_camera.connect()
            if not self.ip_camera.is_open:
                self.logger.error(f"Failed to open stream: {self.ip_camera.ip_address}")
                status = "OpenFailed"
                self.close()
                return

            first_frame_for_security = None
            frame_counter = 0
            self.no_frame_count = 0

            while self.ip_camera.is_open:
                if self.shutdown_event and self.shutdown_event.is_set():
                    self.logger.info("Shutdown event detected. Exiting process loop.")
                    status = "Shutdown"
                    break

                ret, frame = self.ip_camera.capture.read()

                if frame is None:
                    self.no_frame_count += 1
                    if self.no_frame_count >=10:
                        break
                    else:
                        continue

                if not ret or frame is None:
                    self.logger.warning("Frame read failed or no more frames. Exiting.")
                    status = "ReadFail"
                    break

                # 2) Rotation (unchanged)
                if self.rotation and self.rotation != 0:
                    try:
                        if self.rotation == 90:
                            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        elif self.rotation == 180:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        elif self.rotation == 270:
                            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    except cv2.error as e_rotate:
                        self.logger.error(
                            f"[{self.device_name}] Error rotating frame: {e_rotate}. Skipping frame.",
                            exc_info=True
                        )
                        continue  # skip this problematic frame

                # 3) Initialize security module on the very first frame
                if first_frame_for_security is None:
                    self._initialize_security_module_if_needed(frame)
                    first_frame_for_security = frame

                frame_counter += 1

                # 4) Time-based skipping logic
                now = datetime.datetime.now(datetime.timezone.utc)

                if self._last_process_time is None:
                    # First frame we process immediately
                    should_process = True
                else:
                    elapsed = now - self._last_process_time
                    should_process = (elapsed >= self.frame_interval)

                if not should_process:
                    # We are “too early” to run predict, so just save raw (if needed) and continue
                    if self.save_raw_video and self.raw_video_writer:
                        self.raw_video_writer.write_frame(frame)
                    continue

                # 5) It’s time to process a frame:
                self._last_process_time = now
                current_time = now
                frame_number = frame_counter

                # Perform inference + tracking
                _ = self.predict(frame, current_time, frame_number)

                # 6) (Optional) If you still want to write processed video frames:
                # if self.processed_video_writer:
                #     self.processed_video_writer.write_frame(frame)

            # end while

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt caught in process().")
            status = "KeyboardInterrupt"

        except Exception as e:
            self.logger.error(f"Error in process(): {e}", exc_info=True)
            status = f"Error:{e}"

        finally:
            self.logger.info(f"Cleaning up after process() with status: {status}")
            if self.ip_camera and self.ip_camera.is_open:
                self.ip_camera.close()
            if self.raw_video_writer:
                self.raw_video_writer.finish_clip(status=status)
            if self.processed_video_writer:
                self.processed_video_writer.finish_clip(status=status)
            if self.metadata_handler:
                self.metadata_handler.close()
            self.close()
            self.logger.info(f"Resources cleaned up for camera '{self.device_name}' (live mode).")


    # -------------------------------------------------------------------------
    # CLEANUP: destroy TRT model, free CUDA memory, garbage collect
    # -------------------------------------------------------------------------
    def close(self):
        """
        Call this to free GPU resources and destroy the TRT model. After this, no inference is possible.
        """
        if self._is_closed:
            return

        self.logger.info(f"Closing Detector resources for '{self.device_name}'.")

        # 1) Destroy TRT model if present
        if self.trt_model is not None:
            self.logger.info("Destroying TRTInference model...")
            try:
                self.trt_model.destroy()
                self.logger.info("TRTInference model destroyed.")
            except Exception as e:
                self.logger.warning(f"Exception while destroying TRTInference: {e}", exc_info=True)
            finally:
                self.trt_model = None

        # 2) Clear GPU cache & run garbage collector
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("CUDA cache cleared.")
            import gc
            gc.collect()
            self.logger.info("Python garbage collector invoked.")
        except Exception as e:
            self.logger.warning(f"Exception during CUDA cache clear / GC: {e}", exc_info=True)

        self._is_closed = True

    def __del__(self):
        """
        Ensure resources are freed if object is garbage-collected without explicit close().
        """
        try:
            if not self._is_closed:
                self.close()
        except Exception:
            pass
