# -*- coding: utf-8 -*-
'''
Created on 2024-08-01 15:41
Project: sentinel-attendance
@author: Aryan Sinha
@email: arison.aryan@gmail.com
'''

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import datetime
import cv2
import os
import torch
import asyncio
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from src.database.user_handler import UserHandler

class FaceEmbeddingExtractor:
    def __init__(self, device):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_face_embeddings(self, image, bbox):
        x1, y1, x2, y2 = [int(x) for x in bbox]
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        face = np.transpose(face, (2, 0, 1))
        face = torch.from_numpy(face).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            embedding = self.model(face).cpu().numpy()[0]
        return embedding
            


class YOLOFaceDetector:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.device = device

    def detect_faces(self, frame):
        results = self.model.predict(frame)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        return bboxes, confidences


class FaceRegistration:
    def __init__(self):
        self.config  = {
            "yolo_model_path": "src/models/yolov8l_100e.pt",
            "results_directory": "results",
            "frames_directory": "results/frames",
            "min_cosine_dist": 0.009,
            "camera_source": "rtsp://NeophyteTapoCam3:neoVision@192.168.29.197/stream1"
        }
        self.results_directory = self.config["results_directory"]
        self._ensure_directory(self.results_directory)
        self.embeddings_dir = os.path.join(self.results_directory, "embeddings")
        self._ensure_directory(self.embeddings_dir)
        self.frames_directory = self.config["frames_directory"]
        self.frames_dir = os.path.join(self.results_directory, "frames")
        self._ensure_directory(self.frames_dir)
        self.camera_source = self.config["camera_source"]
        self.min_cosine_dist = self.config.get("min_cosine_dist", 0.009)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user = None
        self.face_detector = YOLOFaceDetector(self.config["yolo_model_path"], self.device)
        self.embedding_extractor = FaceEmbeddingExtractor(self.device)
        self.embeddings_storage = {}
        self.count_storage = {}
        self.required_embeddings = 10
        self.captured_embeddings = 0
        # self.face_embeddings = []
    
    def _ensure_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def set_user_details(self, user_info):
        self.user = user_info
        name = f"{user_info.firstname}_{user_info.lastname}"
        embeddings_directory = os.path.join(self.embeddings_dir, name)
        os.makedirs(embeddings_directory, exist_ok=True)
        return embeddings_directory
    
    def is_registration_complete(self):
        return self.captured_embeddings >= self.required_embeddings

    def is_embedding_varied(self, new_embedding, embeddings_list):
        for emb in embeddings_list:
            if cosine(new_embedding, emb) < self.min_cosine_dist:
                return False
        return True
    
    def process_frame(self, frame, face_embedding, name, phone_number):
        if face_embedding is not None:
            stored_embeddings = self.embeddings_storage.get(name, [])
            if self.is_embedding_varied(face_embedding, stored_embeddings):
                self.embeddings_storage.setdefault(name, []).append(face_embedding)
                count = self.count_storage.get(name, 0)
                self.count_storage[name] = count + 1
                embeddings_directory = os.path.join(self.embeddings_dir, name)
                os.makedirs(embeddings_directory, exist_ok=True)
                embeddings_filename = f"{name}_{phone_number}_embeddings_{count}.npy"
                embeddings_filepath = os.path.join(embeddings_directory, embeddings_filename)
                np.save(embeddings_filepath, face_embedding)
                print(f"Stored {embeddings_filepath}")

                frames_directory = os.path.join(self.frames_dir, name)
                os.makedirs(frames_directory, exist_ok=True)
                frame_filename = f"{name}_{phone_number}_frame_{count}.jpg"
                frame_filepath = os.path.join(frames_directory, frame_filename)
                cv2.imwrite(frame_filepath, frame)
                print(f"Stored {frame_filepath}!!!!")
                return embeddings_filepath
            else:
                print("Embedding not varied enough, skipping storage!!!!")
        return None
    
    def load_embeddings_from_directory(self, directory):
        embeddings = []
        labels = []
        for person_name in os.listdir(directory):
            person_dir = os.path.join(directory, person_name)
            if os.path.isdir(person_dir):
                for file_name in os.listdir(person_dir):
                    if file_name.endswith('.npy'):
                        file_path = os.path.join(person_dir, file_name)
                        embedding = np.load(file_path)
                        embeddings.append(embedding)
                        labels.append(person_name)
        return np.array(embeddings), np.array(labels)

    async def register_user_from_camera(self, name, phone_number):
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            raise ValueError(f"Unable to open camera source: {self.camera_source}")
        # count = 0
        last_embedding_path = None
        try:
            while not self.is_registration_complete():
                ret, frame = cap.read()
                if not ret:
                    break

                bboxes, confidences = self.face_detector.detect_faces(frame)
                x_min, y_min = 150.65,297.02
                x_max, y_max = 647.65,963.66
                for bbox, confidence in zip (bboxes, confidences):
                    x, y, w, h = bbox
                    x_end = w
                    y_end = h
                    if (x >= x_min and x_end <= x_max and y >= y_min and y_end <= y_max and confidence > 0.80):
                        face_embedding = self.embedding_extractor.get_face_embeddings(frame, bbox)
                        last_embedding_path = self.process_frame(frame, face_embedding, name, phone_number)
                        if last_embedding_path:
                            self.captured_embeddings += 1

                # Display the frame with bounding boxes
                for bbox, confidence in zip(bboxes, confidences):
                    x1, y1, x2, y2 = [int(x) for x in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            
                _, frame_bytes = cv2.imencode('.jpg', frame)
                frame_bytes = frame_bytes.tobytes()
                # print("FRAME BYTES", frame_bytes)
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
                await asyncio.sleep(0.1)
        except GeneratorExit:
            print("Stream closed gracefully.")
        finally:
            print("Stream ended.")
            cap.release()
            
        # return last_embedding_path

