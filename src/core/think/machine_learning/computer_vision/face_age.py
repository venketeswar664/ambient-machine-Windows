import os
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class FaceDetector:
    """
    TensorFlow-based face detector using a SavedModel.
    Provides a simple API to detect and crop faces from an image.
    """
    def __init__(self, model_dir: str, score_threshold: float = 0.5):
        print(f"[FaceDetector] Loading TF model from {model_dir}")
        self.model = tf.saved_model.load(model_dir)
        self.score_threshold = score_threshold

    def _run_inference(self, image: np.ndarray):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)[tf.newaxis, ...]
        outputs = self.model(tensor)
        num = int(outputs['num_detections'][0])
        boxes = outputs['detection_boxes'][0].numpy()[:num]
        scores = outputs['detection_scores'][0].numpy()[:num]
        return boxes, scores

    def detect_faces(self, image: np.ndarray):
        """
        Detects faces in a BGR image and returns crops and confidences.

        Args:
            image: np.ndarray (H x W x 3)

        Returns:
            List of tuples: (face_crop: np.ndarray, score: float)
        """
        h, w = image.shape[:2]
        boxes, scores = self._run_inference(image)
        results = []
        for box, score in zip(boxes, scores):
            if score < self.score_threshold:
                continue
            ymin, xmin, ymax, xmax = box
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            # Clip coords
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2].copy()
            results.append((crop, float(score)))
        return results


class AgeGenderClassifier:
    """
    TorchScript-based age and gender classifier.
    Accepts in-memory face crops and returns labels and confidences.
    """
    def __init__(self, model_path: str, image_size: int = 224, batch_size: int = 8, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AgeGenderClassifier] Loading model on {self.device} from {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_names = [
            'adult_female', 'adult_male', 'elderly_female', 'elderly_male',
            'kid_female', 'kid_male', 'teen_female', 'teen_male'
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def infer_from_crops(self, crops: list) -> list:
        """
        Run age-gender inference on a list of BGR face crops.

        Args:
            crops: List of np.ndarray face images (BGR format).

        Returns:
            List of tuples: (label: str, confidence: float)
        """
        # Preprocess and convert to tensor list
        tensors = []
        for img in crops:
            # BGR to RGB, then PIL
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = self.transform(pil)
            tensors.append(tensor)
        if not tensors:
            return []
        # Batch
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size], dim=0).to(self.device)
                outputs = self.model(batch)
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    outputs = outputs[0]
                probs = F.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)
                for p, c in zip(preds.cpu().numpy(), confs.cpu().numpy()):
                    label = self.class_names[int(p)]
                    all_preds.append((label, float(c)))
        return all_preds

# Example usage:
# fd = FaceDetector('/path/to/tf_saved_model')
# faces = fd.detect_faces(frame)
# ag = AgeGenderClassifier('/path/to/age_gender.pt')
# results = ag.infer_from_crops([crop for crop, _ in faces])
