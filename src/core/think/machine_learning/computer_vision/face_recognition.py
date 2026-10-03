# -*- coding: utf-8 -*-
'''
Created on 2024-08-02 11:53
Project: ambient-machine
@author: Aryan Sinha
@email: arison.aryan@gmail.com
'''

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from ultralytics import YOLO
from src.core.modules.module import Module

class YOLOFaceDetector:
    def __init__(self, config):
        self.model = YOLO(config['model_path'])
        self.device = config['device']

    def detect_faces(self, input_dict):
        frame = input_dict["frame"]
        results = self.model.predict(frame)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        return {"bboxes": bboxes}

class FaceEmbeddingExtractor:
    def __init__(self, config):
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(config['device'])
        self.device = config['device']

    def extract_embedding(self, input_dict):
        frame = input_dict["frame"]
        bboxes = input_dict["bboxes"]
        embeddings = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face = np.transpose(face, (2, 0, 1))
            face = torch.tensor(face).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                embedding = self.model(face).cpu().numpy()
            embeddings.append(embedding)
        return {"embeddings": np.array(embeddings)}

class SVMClassifier:
    def __init__(self, config):
        with open(config['model_path'], 'rb') as f:
            self.model = pickle.load(f)
        with open(config['label_encoder_path'], 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.confidence_threshold = config['confidence_threshold']

    def predict_label(self, input_dict):
        embeddings = input_dict["embeddings"]
        labels = []
        confidences = []
        for embedding in embeddings:
            probas = self.model.predict_proba(embedding)
            max_proba = np.max(probas)
            if max_proba >= self.confidence_threshold:
                predicted_label = self.label_encoder.inverse_transform([np.argmax(probas)])[0]
            else:
                predicted_label = "Unknown"
            labels.append(predicted_label)
            confidences.append(max_proba)
        return {"labels": labels, "confidence": confidences}

class LiveFaceRecognition(Module):
    def __init__(self, config):
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.face_detector = YOLOFaceDetector(config)
        self.embedding_extractor = FaceEmbeddingExtractor(config)
        self.classifier = SVMClassifier(config)
        self.cap = None


    def process_frame(self, frame, shared_dict=None):
        """
        Process a single frame: detect faces, extract embeddings, and predict labels.

        Args:
            frame (np.ndarray): The input image frame to process.
            shared_dict (dict, optional): A dictionary to store output data. Defaults to None.

        Returns:
            dict: A dictionary containing the detected labels, confidences, and bounding boxes.
        """
        # Initialize the input dictionary with the given frame
        input_dict = {"frame": frame}

        # Perform face detection
        output_dict = self.face_detector.detect_faces(input_dict)

        # Extract embeddings for the detected faces
        output_dict.update(self.embedding_extractor.extract_embedding(output_dict))

        # Predict labels for the embeddings
        output_dict.update(self.classifier.predict_label(output_dict))

        # Extract results
        labels = output_dict["labels"]
        confidences = output_dict["confidence"]
        bboxes = output_dict["bboxes"]

        # Draw results on the frame
        for (label, confidence, box) in zip(labels, confidences, bboxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the frame (optional, can be commented out if not needed)
        cv2.imshow('Processed Frame', frame)
        cv2.waitKey(0)  # Press any key to close the window

        # Optionally store the results in the shared_dict
        if shared_dict is not None:
            shared_dict[self.device_name]["data"] = output_dict

        return output_dict


