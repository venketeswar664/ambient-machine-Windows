# -*- coding: utf-8 -*-
'''
Created on 2024-09-24 15:30
Project: ambient-machine
@author: Soham Ghosh
@email: soham.ghosh@neophyte.ai
'''


'''
 "embedding_extractor": {
        "name": "Face Embedding Extractor",
        "module": "src/core/machine_learning/computer_vision/embedding_extractor.py",
        "config": {
          "model_name": "InceptionResnetV1",
          "pretrained": "vggface2",
          "device": "cuda"
        },
        "input": ["frame", "bboxes", "confidences"],
        "output": ["embeddings", "bboxes", "frame"],
        "process": ["process"]
      },'''

import cv2
import numpy as np

import torch

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from src.core.modules.module import Module
from src.utils import plot

class FaceEmbeddingExtractor(Module):
    def __init__(self, _config):
        """
        Initialize the FaceEmbeddingExtractor with configuration settings.

        Args:
            _config (dict): Configuration dictionary containing model and device settings.
        """ 
        Module.__init__(self, _config)
        self.config = _config
        self.img_path = self.config["temp_frame"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

        # # Move the model to the GPU if available
        # if self.device.type == 'cuda':
        #     self.model = self.model.to(self.device)


    # Function to load and preprocess the image
    def load_and_preprocess_image(self):
        img = image.load_img(self.img_path, target_size=(299, 299))  # InceptionResNetV2 uses 299x299
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    # Function to extract embeddings
    def extract_embeddings(self):
        preprocessed_image = self.load_and_preprocess_image()
        embeddings = self.model.predict(preprocessed_image)
        return embeddings

    def process(self, input_dict) -> dict:
        """
        Extract embeddings from the given frame and bounding boxes.

        Args:
            input_dict (dict): Input dictionary containing "frame" and "bboxes".

        Returns:
            dict: Output dictionary containing "embeddings".
        """
        frame = input_dict["frame"]
        bboxes = input_dict["bboxes"]
        confidences = input_dict["confidences"]
        fps_frame = input_dict["camera_fps"]
        embeddings = []
        if len(bboxes) > 0:
            for bbox, confidence in zip(bboxes, confidences):
                print(confidence)
                if confidence>0.7:
                    x1, y1, x2, y2 = map(int, bbox)
                    face = frame[y1:y2, x1:x2]
                    # face = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(self.img_path, face)
                    with torch.no_grad():
                        embedding = self.extract_embeddings()
                    embeddings.append(embedding)

        return {"embeddings": np.array(embeddings), "bboxes": bboxes, "confidences":confidences, "camera_fps":fps_frame, "frame":frame}
