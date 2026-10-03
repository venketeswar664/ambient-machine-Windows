# -*- coding: utf-8 -*-
'''
Created on 26-July-2024 17:55
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''

"""
_config = {
            "name": "person detector",
            "module": "src/core/machine_learning/computer_vision/detector.py",
            "config": {
                "model_path": "./src/models/Augmented_Digital_30Mar_2024.engine",
                "tracker_path": "src/configs/machine_learning/computer_vision/tracker/botsort_with_reid.yaml",
                "plot": True
            },
            "process": {

            }

"""

from ultralytics import YOLO
import datetime
from shapely.geometry import Point, Polygon, LineString
from src.core.modules.module import Module
from src.utils import plot


class Detector(Module):
    def __init__(self, _config) -> None:
        Module.__init__(self, _config)
        self.config = _config
        self.model_path = self.config["model_path"]
        self.tracker_path = self.config["tracker_path"]
        self.plot = self.config["plot"]
        self.conf = self.config["conf"]
        self.class_IDS = self.config["class_ids"]
     

        self.load_model()

    
    def process(self, input_dict) -> dict:

        frame = input_dict["frame"]
        time_stamp = datetime.datetime.now(datetime.timezone.utc)()

        labels, masks, boxes, track_ids, confidences, frame = self.predict(frame)

        if track_ids:
            output_dict={
                "labels": labels,
                "masks":masks,
                "boxes":boxes,
                "track_ids":track_ids,
                "confidence":confidences,
                "time_stamp":time_stamp,
                "frame":frame
            }
            

            if self.plot:
         
                # plot track id and its positions
                frame = plot.plot_point_and_trackid(frame, boxes, track_ids, labels)

        if self.plot:
            # plot shapes
            frame = plot.plot_shapes(frame, self.polygons)
            output_dict["frame"] = frame




        return output_dict

    def load_model(self):
        self.model = YOLO(self.model_path, task="detect")

    
    def predict(self, frame):
        """
        Method to apply yolo detector on the frame

        Args:
            frame (numpy array): frame

        Returns:
        keypoints: human body keypoints based on yolo output
        boxes: human body bbox
        track_ids (https://docs.ultralytics.com/modes/track/) : track ids of detected human 
        """


        masks = None
        boxes = None
        track_ids = None
        labels = None
        confidence = None

        # results = self.pose_model.track(_frame, persist=True, verbose=False, conf = conf_level, classes = class_IDS, device = 0, embed=[20])
        results = self.model.track(frame, persist=True, verbose=False, conf = self.conf, \
                                   classes = self.class_IDS, device = 0, tracker=self.tracker_path)

    
        if results[0].boxes.id != None:
            # Get the keypoints, boxes and track
            # masks = results[0].masks.xy 
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            labels = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            print(boxes)

        if self.plot:
            # Visualize the results on the frame
            # colors = {"cat": "red", "dog": "blue"}
            frame = results[0].plot(conf=False, labels=False, boxes=False, masks=False)

        return labels, masks, boxes, track_ids, confidences, frame