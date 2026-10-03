import cv2
import datetime
import time
import numpy as np
from mongoengine import connect
from ultralytics import YOLO
from mongoengine import Document, ReferenceField, BooleanField, DateTimeField, StringField

# Security Schema
class Security(Document):
    camera = StringField(required=True)
    time_stamp = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    lights = BooleanField(required=True)
    camera_tampering = BooleanField(required=True)
    
    meta = {
        'collection': 'security'
    }

# Fire & Smoke Detector
class FireSmokeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = getattr(self.model, 'names', {0: 'Fire', 1: 'Smoke'})

    def detect_from_image(self, image, conf_threshold=0.25):
        results = self.model(image, conf=conf_threshold)[0]
        detection_results = {'fire_detected': False, 'smoke_detected': False}
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if conf >= conf_threshold:
                class_name = self.class_names[cls_id].lower()
                if 'fire' in class_name:
                    detection_results['fire_detected'] = True
                if 'smoke' in class_name:
                    detection_results['smoke_detected'] = True
        return detection_results
        
# Camera Tampering Detector
class EdgeDifferenceDetector:
    def __init__(self, template_image):
        self.template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    def check_difference(self, current_frame, threshold=30):
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        difference = np.abs(self.template_gray.astype(float) - gray_current.astype(float))
        mean_difference = np.mean(difference)
        return mean_difference > threshold

    
# Light Detector
class LightDetector:
    def __init__(self, brightness_threshold=80):
        self.brightness_threshold = brightness_threshold

    def is_light_on(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness >= self.brightness_threshold

# Security Monitoring Module
class SecurityModule:
    def __init__(
        self,
        camera_id,
        template_frame,
        detect_light=True,
        detect_edge=True
    ):
        self.camera_id = camera_id
        # Detection flags
        self.detect_light = detect_light
        self.detect_edge = detect_edge
        
        # Initialize detectors
        # self.fire_smoke_detector = FireSmokeDetector(fire_smoke_model_path) if self.detect_fire_smoke else None
        self.light_detector = LightDetector() if self.detect_light else None
        self.edge_detector = EdgeDifferenceDetector(template_frame) if self.detect_edge else None
        
        # Timestamps for throttling
        self.last_fire_smoke_check = time.time()
        self.last_tampering_check = time.time()
        self.last_light_check = time.time()
        self.last_db_push = time.time()
        
        # State
        self.lights_on = True
        self.camera_tampered = False
        # self.smoke_detected = False
        # self.fire_detected = False
        
        print(f"Initialized SecurityModule for camera: {self.camera_id}")

    def process_frame(self, frame):
        current_time = time.time()

        # Fire & Smoke Detection (Every 60 seconds)
        # if self.detect_fire_smoke and (current_time - self.last_fire_smoke_check >= 60):
        #     detection = self.fire_smoke_detector.detect_from_image(frame)
        #     self.smoke_detected = detection['smoke_detected']
        #     self.fire_detected = detection['fire_detected']
        #     self.last_fire_smoke_check = current_time

        # Camera Tampering Detection (Every 300 seconds)
        if self.detect_edge and (current_time - self.last_tampering_check >= 300):
            self.camera_tampered = self.edge_detector.check_difference(frame)
            self.last_tampering_check = current_time

        # Light Detection (Every 60 seconds)
        if self.detect_light and (current_time - self.last_light_check >= 60):
            self.lights_on = self.light_detector.is_light_on(frame)
            self.last_light_check = current_time

        # Push to DB (Every 60 seconds)
        if current_time - self.last_db_push >= 60:
            self.push_to_db()
            self.last_db_push = current_time

    def push_to_db(self):
        try:
            security_doc = Security(
                camera=self.camera_id,
                lights=self.lights_on,
                camera_tampering=self.camera_tampered
            )
            security_doc.save()
            print("Security data pushed to DB:", security_doc.to_json())
        except Exception as e:
            print("Error pushing security data to DB:", e)

