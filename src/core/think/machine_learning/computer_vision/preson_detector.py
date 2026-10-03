# src/core/think/machine_learning/computer_vision/detector.py

import cv2
import datetime
from ultralytics import YOLO
from src.core.machine import Machine
from src.utils import plot

class PersonDetector(Machine):
    """
    PersonDetector machine that uses YOLO model to detect and track persons in frames.
    """

    def __init__(self, machine_name, state_manager):
        super().__init__(machine_name, state_manager)
        config = self.config.get('config', {})

        # Load configurations
        self.device = config.get('device', 'cpu')
        self.model_path = config.get('model_path')
        self.tracker_path = config.get('tracker_path')
        self.plot = config.get('plot', False)
        self.conf = config.get('conf', 0.6)
        self.class_ids = config.get('class_ids', [0])  # Default to person class
        self.polygons = []
        # Load the model
        self.model = YOLO(self.model_path).to(self.device)
        self.ip_camera = None

    def get_data(self):
        """
        Retrieves the 'ip_camera' instance from the shared data_dict.
        """
        input_data = super().get_data()

        if "ip_camera" in input_data and self.ip_camera is None:
            self.ip_camera = input_data.get('ip_camera')
            self.polygons = self.ip_camera.polygons
        return input_data

    def process(self, input_data):
        """
        Detects and tracks persons in the current frame from the IP camera.
        """
        if not self.ip_camera:
            if self.logger:
                self.logger.error("No 'ip_camera' instance available.")
            return {}

        frame = self.ip_camera.get_frame()

        if frame is None or isinstance(frame, str):
            if self.logger:
                self.logger.warning("No frame received from IP camera.")
            if isinstance(frame, str):
                self.logger.warning(f"Reason: {frame}")
            return {}

        # Perform detection and tracking
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device,
            tracker=self.tracker_path
        )

        output_dict = {}
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            labels = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            time_stamp = datetime.datetime.now(datetime.timezone.utc)().isoformat()

            output_dict = {
                "labels": labels,
                "boxes": boxes,
                "track_ids": track_ids,
                "confidences": confidences,
                "time_stamp": time_stamp,
                "frame": frame
            }

            if self.plot:
                # Plot track IDs and positions
                frame = plot.plot_point_and_trackid(frame, boxes, track_ids, labels)
                output_dict["frame"] = frame

            if len(self.polygons):
                

        else:
            if self.logger:
                self.logger.info("No persons detected.")

        if self.plot:
            # Plot shapes if needed (e.g., polygons)
            if hasattr(self, 'polygons') and len(self.polygons) > 0:
                frame = plot.plot_shapes(frame, self.polygons)
                output_dict["frame"] = frame

        if self.logger and output_dict:
            num_persons = len(output_dict.get('track_ids', []))
            self.logger.info(f"Detected and tracked {num_persons} persons.")

        return output_dict

    def send_data(self, processed_data):
        """
        Sends the detection and tracking results to the shared data_dict.
        """
        super().send_data(processed_data)
