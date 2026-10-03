# src/core/think/machine_learning/computer_vision/embedding_extractor_resnetV1.py

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from src.core.machine import Machine
from torchvision import transforms

class FaceEmbeddingExtractor(Machine):
    """
    FaceEmbeddingExtractor machine that extracts embeddings from detected faces.
    """

    def __init__(self, machine_name, state_manager):
        super().__init__(machine_name, state_manager)
        config = self.config.get('config', {})
        self.padding = 60
        self.device = torch.device(config.get('device', 'cpu'))
        model_name = config.get('model_name', 'InceptionResnetV1')
        pretrained = config.get('pretrained', 'vggface2')
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted for RGB images
        ])

    def process(self, input_data):
        """
        Extracts embeddings from the faces detected in the frame.
        """
        frame = input_data.get('frame')
        bboxes = input_data.get('bboxes', [])
        confidences = input_data.get('confidences', [])

        embeddings = []
        processed_bboxes = []
        processed_frame = frame

        if frame is None or len(bboxes) == 0:
            if self.logger:
                self.logger.warning("No frame or bounding boxes to process.")
            return {
                "embeddings": np.array(embeddings),
                "bboxes": processed_bboxes,
                "frame": processed_frame
            }

        for bbox, confidence in zip(bboxes, confidences):
            if confidence > 0.7:
                x1, y1, x2, y2 = map(int, bbox)
                y1_padded = max(0, y1 - self.padding)
                y2_padded = min(frame.shape[0], y2 + self.padding)
                x1_padded = max(0, x1 - self.padding)
                x2_padded = min(frame.shape[1], x2 + self.padding)

                # Crop the face region with padding
                face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                face_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                face_image = Image.fromarray(face_image)
                img_tensor = self.transform(face_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    embedding = self.model(img_tensor)
                embeddings.append(embedding.cpu().numpy())
                processed_bboxes.append(bbox)

        processed_data = {
            "embeddings": np.array(embeddings),
            "bboxes": processed_bboxes,
            "frame": processed_frame
        }

        if self.logger:
            self.logger.info(f"Extracted embeddings for {len(embeddings)} faces.")

        return processed_data

    def send_data(self, processed_data):
        """
        Sends the embeddings and related data to the shared data_dict.
        """
        super().send_data(processed_data)

