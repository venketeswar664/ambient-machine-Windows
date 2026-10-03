import cv2
from ultralytics import YOLO
import datetime

import sys

sys.path.append(".")

import numpy as np
from tqdm import tqdm
import concurrent.futures
import logging
from shapely.geometry import Point, Polygon, box
from src.utils import plot  # Assuming you have a plot utility
from src.database.schemas.zone_schema import Zone
from src.database.database import Database
from src.database.metadata_handler import MetadataHandler  # Ensure correct path
import os

import json
import argparse
import threading
import mongoengine as db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HumanDetector:
    """
    HumanDetector machine that uses a YOLO model to detect humans in frames.
    """

    def __init__(self, config):
        # Load configurations
        self.device = config.get('device', 'cuda:0')
        self.model_path = config.get('model_path')
        self.tracker_path = config.get('tracker_path')
        self.plot = config.get('plot', True)
        self.conf = config.get('conf', 0.6)
        self.class_ids = config.get('class_ids', [0,1])  # Default to person class

        self.video_path = config.get("video_path", "")
        self.device_name = config.get("device_name", "")
        # Load the model
        self.model = YOLO(self.model_path, task="detect").to(self.device)
        self.model.tracker = self.tracker_path

        self.conf_threshold = config.get('conf_threshold', 0.4)

        # Initialize database connection
        self.database = Database()
        self.database.connect_db()

        # Load zones (if provided)
        self.zone_dict = config.get('zone_dict', {})
        self.polygons = [Polygon(zone) for zone in self.zone_dict.values()] if self.zone_dict else []

        # Initialize MetadataHandler
        self.metadata_handler = MetadataHandler(self.video_path, self.device_name)

    def _get_center(self, bbox):
        """
        Calculate the center of the bounding box.
        bbox is in the format [x1, y1, x2, y2].
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2  # Corrected y_center calculation
        return (x_center, y_center)

    def predict(self, frame):
        """
        Method to apply YOLO detector on the frame.

        Returns:
            labels, masks, boxes, track_ids, confidence, frame
        """
        masks = None
        boxes = None
        track_ids = None
        labels = None
        confidence = None

        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device
        )

        if len(results) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            labels = results[0].boxes.cls.int().cpu().numpy()
            confidence = results[0].boxes.conf.cpu().numpy()

        if self.plot:
            # Visualize the results on the frame (returns RGB)
            frame = results[0].plot(conf=False, labels=True, boxes=True, masks=False)

            # Convert RGB to BGR for OpenCV
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 # Optionally plot zones if zone_dict is not empty
            if self.zone_dict:
                frame = plot.plot_shapes(frame, self.polygons)

                # Validate the frame after plotting shapes
                if frame is None or not isinstance(frame, np.ndarray):
                    logging.error("plot_shapes returned an invalid frame.")
                else:
                    logging.debug(f"Frame shape after plot_shapes: {frame.shape}")

            # # Add track_id labels on top of each bounding box
            # if track_ids is not None and boxes is not None:
            #     for i, box_coords in enumerate(boxes):
            #         track_id = track_ids[i]
            #         label = labels[i] if labels is not None else -1  # Default label if not present

            #         # Define the position for the track_id label
            #         x1, y1, x2, y2 = box_coords
            #         text_position = (int(x1), int(y1) - 10)  # 10 pixels above the top-left corner

            #         # Prepare the text to display
            #         text = f"ID: {track_id}"

            #         # Choose font, scale, color, and thickness
            #         font = cv2.FONT_HERSHEY_SIMPLEX
            #         font_scale = 0.5
            #         color = (0, 255, 0)  # Green color for visibility
            #         thickness = 2

            #         # Ensure the text is within frame boundaries
            #         text_position = (
            #             max(text_position[0], 0),
            #             max(text_position[1], 0)
            #         )

            #         # Put the text on the frame
            #         cv2.putText(frame, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)


        return labels, masks, boxes, track_ids, confidence, frame

    def process(self):
        """
        Detects humans in the current frame from the video file at self.video_path,
        and updates the timestamp based on the frame number and FPS.
        """

        # Initialize the video capture object
        cap = cv2.VideoCapture(self.video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            logging.error(f"Couldn't open video file at {self.video_path}")
            return
        logging.info(f"Opened video file: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
        logging.info(f"Total frames: {total_frames}, FPS: {fps}")

        progress_bar = tqdm(total=total_frames, desc='Processing Video', unit='frame')

        frame_number = 0  # Initialize the frame number

        # Get the initial timestamp
        initial_time_stamp = datetime.datetime.now(datetime.timezone.utc)()

        # Initialize VideoWriter if plotting is enabled
        if self.plot:
            # Extract the base filename (without extension)
            base_filename = os.path.splitext(os.path.basename(self.video_path))[0]
            # Construct output path with the "_yolo_plot" suffix
            output_dir = os.path.dirname(self.video_path)
            output_path = os.path.join(output_dir, f"{base_filename}_yolo_plot.mp4")

            # Create the directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"Created directory: {output_dir}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can try 'XVID' or 'H264' if 'mp4v' fails
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            if not out.isOpened():
                logging.error(f"Failed to open VideoWriter with path {output_path}")
                return
            logging.info(f"Initialized VideoWriter with path: {output_path}")

        try:
            while True:
                ret, frame = cap.read()  # Read a frame from the video
                if not ret:  # If the frame was not successfully read (e.g., end of video)
                    logging.info("Reached the end of the video or failed to read frame.")
                    break

                frame_number += 1  # Increment frame number

                # Calculate the time in seconds for this frame
                time_in_seconds = frame_number / fps
                # Calculate the new timestamp by adding the time in seconds to the initial timestamp
                time_stamp = initial_time_stamp + datetime.timedelta(seconds=time_in_seconds)

                # Process the frame (e.g., run detection)
                labels, masks, boxes, track_ids, confidence, processed_frame = self.predict(frame)
                # logging.debug(f"Frame {frame_number}: Detected labels: {labels}")

                if track_ids is not None:
                    zones_track_id = {}
                    # Only process zones if zone_dict is not empty
                    if self.zone_dict:
                        zones_track_id = self.zones_iou_check(labels, boxes, track_ids, confidence)

                    # Store the processed data
                    processed_data = {
                        "time_stamp": time_stamp,  # Pass datetime object
                        "frame_number": frame_number,
                        "labels": labels,
                        "masks": masks,
                        "boxes": boxes.tolist(),
                        "track_ids": track_ids,
                        "confidences": confidence,
                        "zones_track_id": zones_track_id,
                        "frame": processed_frame
                    }

                    # Process metadata and save
                    self.metadata_handler.process(processed_data)

                # Write the frame to output video if plotting
                if self.plot:
                    # Ensure frame size consistency
                    if (processed_frame.shape[1], processed_frame.shape[0]) != (frame_width, frame_height):
                        logging.warning(
                            f"Frame size mismatch at frame {frame_number}. "
                            f"Expected ({frame_width}, {frame_height}), "
                            f"got ({processed_frame.shape[1]}, {processed_frame.shape[0]})"
                        )
                        # Resize the frame to match the VideoWriter's expected size
                        processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

                    out.write(processed_frame)
                    if frame_number % 100 == 0:
                        logging.info(f"Written frame {frame_number} to output video.")

                progress_bar.update(1)  # Update the progress bar

        except Exception as e:
            logging.error(f"Error in HumanDetector: {e}")

        finally:
            # Clean up
            progress_bar.close()
            cap.release()  # Release the video capture object
            logging.info("Released VideoCapture object.")
            if self.plot:
                out.release()  # Release the VideoWriter object
                logging.info(f"Released VideoWriter and saved video to {output_path}")
            self.metadata_handler.close()  # Shutdown thread pool
            cv2.destroyAllWindows()  # Close all OpenCV windows if any were opened
            logging.info("Cleaned up all resources.")

    def zones_iou_check(self, labels, boxes, track_ids, confidence):
        zones_track_id = {}

        for i, track_id in enumerate(track_ids):
            conf = confidence[i]
            if conf > self.conf_threshold:
                bbox = boxes[i]
                current_position = self._get_center(bbox)
                point = Point(current_position)

                for name, polygon_points in self.zone_dict.items():
                    polygon = Polygon(polygon_points)

                    if polygon.contains(point):
                        if name not in zones_track_id:
                            zones_track_id[name] = []

                        person_bbox = box(*bbox)
                        intersection = polygon.intersection(person_bbox).area
                        union = polygon.union(person_bbox).area
                        iou_value = intersection / union if union != 0 else 0

                        zones_track_id[name].append({
                            "track_id": track_id,
                            "iou": iou_value,
                            "label": labels[i]
                        })

                        logging.debug(f"zones_track_id: {zones_track_id}")

        return zones_track_id


# MetadataHandler and Metadata classes should be in separate files, but included here for completeness


class MetadataHandler:
    def __init__(self, video_path, device_name):
        self.video_path = video_path
        self.device_name = device_name
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Adjust as needed

    def process(self, input_dict) -> dict:
        self.executor.submit(self.push_data, input_dict)
        return {}

    def push_data(self, input_dict) -> dict:
        if input_dict["track_ids"] is not None:
            track_ids = list(input_dict["track_ids"])
            frame_number = int(input_dict["frame_number"])
            time_stamp = input_dict["time_stamp"]
            zones_track_id = input_dict["zones_track_id"]
            labels = list(input_dict["labels"])
            confidence = list(input_dict["confidences"])
            boxes = list(input_dict["boxes"])
            frame = input_dict["frame"]

            metadata = Metadata(
                frame_number=frame_number,
                time_stamp=time_stamp,  # datetime object
                track_ids=track_ids,
                labels=labels,
                confidence=confidence,
                bboxes=boxes,
                video_path=self.video_path,
                device_name=self.device_name,
                zones_track_id=zones_track_id,
            )

            metadata.save()
            if not frame_number % 100:
                logging.info(f"Saved Metadata for frames: {frame_number}")

        return {}

    def close(self):
        self.executor.shutdown(wait=True)


# Define the Metadata document
class Metadata(db.Document):
    frame_number = db.IntField(required=True)
    time_stamp = db.DateTimeField(required=True)
    track_ids = db.ListField(db.IntField(required=True))
    labels = db.ListField(db.IntField(required=True))  # employ = 1 or customer = 0
    bboxes = db.ListField(db.ListField(db.IntField(), required=True))
    confidence = db.ListField(db.FloatField(required=True))

    video_path = db.StringField(required=True)
    device_name = db.StringField(required=True)
    zones_track_id = db.DictField(field=db.ListField(db.DictField()))

    meta = {
        'collection': 'metadata_test',  # Specify the collection name
        'indexes': [
            'time_stamp'
        ]
    }


def main():

    config = {
        "model_path": "/home/soham/projects/trendz/emp_cust_det_20_11_24/runs/detect/train/weights/best.pt",
        "tracker_path": "src/configs/machine_learning/computer_vision/tracker/botsort_with_reid.yaml",
        "device": "cuda",
        "plot": True,
        "conf": 0.6,
        "class_ids": [0, 1],
        "video_path": "/home/pranjal/project/ambient-machine/experiments/samurai/samurai/data/test/customer_exit.mp4",
        "device_name": "my_device",
        "conf_threshold": 0.4,
        "zone_dict": {
            # "entrance": [[100, 200], [150, 200], [150, 250], [100, 250]],
            # "exit": [[300, 400], [350, 400], [350, 450], [300, 450]]
        }
    }

    # Initialize and run detector
    detector = HumanDetector(config)
    detector.process()


if __name__ == '__main__':
    main()
