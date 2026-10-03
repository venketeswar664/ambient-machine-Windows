# src/core/think/manager/predictor.py

import os
import torch
from ultralytics import YOLO

try:
    from ultralytics.nn.modules.block import C3k2  # noqa: F401
except Exception:
    pass


class Predictor:
    """
    Model-only runner:
      - Loads YOLO from .pt or TensorRT .engine.
      - Optional auto-export to .engine via ultralytics.
      - Runs track() and returns (track_ids_dict, frame).
    """
    def __init__(
        self,
        model_path,
        tracker_path,
        device,
        logger,
        conf=0.6,
        class_ids=[0],
        conf_threshold=0.4,
        use_trt=False,
        auto_export_trt=False,
        engine_path=None,
        plot_overlays=False
    ):
        self.logger = logger
        self.model_path = model_path
        self.tracker_path = tracker_path
        self.device = device
        self.conf = conf
        self.class_ids = class_ids
        self.conf_threshold = conf_threshold
        self.plot = plot_overlays

        if not os.path.exists(self.tracker_path):
            raise FileNotFoundError(f"Tracker file not found at {self.tracker_path}")

        # Force CPU device
        self.device = 'cpu'
        self.logger.info("Forcing device to CPU.")
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        if not self.model_path.endswith(".pt"):
            raise ValueError(f"Only .pt models are supported. Received: {self.model_path}")

        self.logger.info(f"Loading YOLO .pt model: {self.model_path} on CPU")
        self.model = YOLO(self.model_path, task="detect")
        self.model.tracker = self.tracker_path
        self.logger.info("Model loaded.")

    def predict(self, frame, current_time, run_time_str):
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

        boxes = results[0].boxes.xyxy.int().cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        labels = results[0].boxes.cls.int().cpu().numpy()
        confidence = results[0].boxes.conf.cpu().numpy()
        class_names = results[0].names

        for idx, tid in enumerate(track_ids):
            updated_track_id = f"{run_time_str}_{tid:05d}"
            track_ids_dict[updated_track_id] = {
                "track_id": updated_track_id,
                "bbox": boxes[idx],
                "label": labels[idx],
                "confidence": confidence[idx],
                "label_name": class_names[labels[idx]],
                "current_time": current_time
            }

        if self.plot:
            frame = results[0].plot(conf=False, labels=True, boxes=True, masks=True)

        return track_ids_dict, frame

    def finish(self):
        self.logger.info("Cleaning up Predictor resources...")
        if hasattr(self, 'model'):
            del self.model
        self.logger.info("Predictor resources released.")
