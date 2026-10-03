import cv2
import numpy as np
from ultralytics import YOLO
from src.core.modules.module import Module
from src.utils import plot

class YoloFaceClassify(Module):
    def __init__(self, _config):
        Module.__init__(self, _config)
        self.config = _config
        self.yolo_model_path = self.config["yolo_classification_model_path"]
        self.classification_confidence_threshold = self.config["confidence_threshold"]
        self.plot = self.config["plot"]
        self.classification_model = YOLO(self.yolo_model_path)
        self.default_time_window = 160
        self.time_window = self.default_time_window
        self.detection_array = []
        self.label_counts = {}
        self.label_confidences = {}

    
    
    def process(self, input_dict) -> str:
        
        output_dict={}
        bboxes = input_dict["bboxes"]
        frame = input_dict["frame"]
        # person_name = "Not_Detected"
        person_confidence = 0
        x_min, y_min = 415, 275
        x_max, y_max = 763, 598
        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        if len(bboxes)!=0:

            for bbox in bboxes:
        
                x1, y1, x2, y2 = bbox

                # print(bbox)
                # if (x1 >= x_min and x2 <= x_max and y1 >= y_min and y2 <= y_max):
                face_frame = frame[y1:y2,x1:x2].copy()
                
                class_results = self.classification_model(face_frame, verbose=False)

                person_cls = class_results[0].names

                person_conf = class_results[0].probs.data.cpu().numpy()
                # print("person_conf: ", person_conf)
                max_index = np.argmax(person_conf)

                person_confidence = person_conf[max_index]
                print("person_confidence after arg max: ",person_confidence)
                
                if person_confidence>0.90:
                    person_name = person_cls[max_index]  
                    classification_conf = (float(person_confidence)*100)
                else:
                    person_name = 'Unknown'
                    classification_conf = 0

                print(person_name)

                
                print("classification_conf: ",classification_conf)
                frame = plot.plot_faces_with_labels(frame, bbox, person_name, classification_conf)
                detection_dict = {
                "labels":person_name,
                "confidences":classification_conf
                }
                self.detection_array.append(detection_dict)
                print("Person detected pushing in detection")
        self.time_window-=1
        if self.time_window <= 0:
            # Reset time_window
            self.time_window = self.default_time_window

            # Process the detection array to find the most frequent label
            for detection in self.detection_array:
                label = detection["labels"]
                confidence = detection["confidences"]
                self.label_counts[label] = self.label_counts.get(label, 0) + 1
                self.label_confidences[label] = max(self.label_confidences.get(label, 0), confidence)
            print("self.label_counts: ",self.label_counts)
            # Find the most frequent label and its max confidence
            if self.label_counts:
                most_frequent_label = max(self.label_counts, key=self.label_counts.get)
                max_confidence = self.label_confidences[most_frequent_label]
            else:
                most_frequent_label = 'Unknown'
                max_confidence = 0

            # Reset detection array and counters
            self.detection_array = []
            self.label_counts = {}
            self.label_confidences = {}

            output_dict = {
                "labels": most_frequent_label,
                "confidences": max_confidence
            }
        print("output_dict: ", output_dict)
        return output_dict
