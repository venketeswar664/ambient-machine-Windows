import cv2
import json
import numpy as np
from ultralytics import YOLO

class QueueDetectionEngine:
    def __init__(self, config_path):
        # Load configuration from the provided config file
        with open(config_path, "r") as f:
            config = json.load(f)
            
        detector_config = config["queue_detector"]
        self.input_config = detector_config["input"]
        self.model_config = detector_config["config"]

        # Initialize parameters from config file
        self.video_path = self.input_config["video_path"]
        self.roi_json_path = self.input_config["roi_json_path"]
        self.model_path = self.model_config["model_path"]
        self.conf_threshold = self.model_config["conf_threshold"]
        self.head_count_threshold = self.model_config["head_count_threshold"]
        self.process_fps = self.model_config["process_fps"]

        # Load head detector model
        self.head_detector = YOLO(self.model_path)
        self.head_detector.conf = self.conf_threshold

        # Load ROI coordinates
        self.load_roi_coordinates()

        # List to store congestion status (only a boolean for each processed frame)
        self.congestion_statuses = []

    def load_roi_coordinates(self):
        with open(self.roi_json_path, "r") as f:
            rois = json.load(f)
        self.queue_roi = (rois["queue"]["x"], rois["queue"]["y"],
                          rois["queue"]["w"], rois["queue"]["h"])

    @staticmethod
    def point_in_roi(point, roi):
        x, y, w, h = roi
        px, py = point
        return (px >= x) and (px <= x + w) and (py >= y) and (py <= y + h)

    def process_frame(self, frame):
        """
        Process a single frame and return the count of heads detected within the ROI.
        """
        queue_head_count = 0
        results = self.head_detector(frame)
        detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

        for box in detections:
            x1, y1, x2, y2 = box[:4]
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if self.point_in_roi(center, self.queue_roi):
                queue_head_count += 1

        return queue_head_count

    def get_fps(self):
        """Retrieve the original FPS of the input video."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30  # Default to 30 FPS if unavailable

    def process_video(self):
        """
        Process the video by sampling frames to match the process_fps specified in the config.
        For every processed frame, determine the congestion status (True/False) based on the head count threshold.
        """
        cap = cv2.VideoCapture(self.video_path)
        video_fps = self.get_fps()
        # Calculate skip_factor to process at the desired process_fps rate
        skip_factor = max(1, int(round(video_fps / self.process_fps)))
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process only every skip_factor-th frame
            if frame_index % skip_factor == 0:
                queue_head_count = self.process_frame(frame)
                congested = queue_head_count > self.head_count_threshold
                # Store only the congestion status (True/False)
                self.congestion_statuses.append(congested)
                print(f"Frame {frame_index}: Congestion status: {congested}")
            frame_index += 1

        cap.release()

def run_queue_detection(config_path, output_path="queue_detection_output.json"):
    """
    Instantiate the QueueDetectionEngine using the provided config file,
    process the video at the specified processing FPS, and store per-frame congestion status.
    The output is a dictionary with a 'congested' key containing a list of booleans.
    """
    processor = QueueDetectionEngine(config_path)
    processor.process_video()

    # Prepare output according to the config's output specification.
    output_result = {
        "congested": processor.congestion_statuses
    }

    # Save the output result into a JSON file for later database insertion
    with open(output_path, "w") as f:
        json.dump(output_result, f, indent=4)

    print("\nProcessing complete. Congestion statuses for each processed frame have been stored.")
    print(f"Output stored in: {output_path}")

# Allow the module to be imported or executed directly.
if __name__ == "__main__":
    default_config_path = "billing_config.json"
    run_queue_detection(default_config_path)
