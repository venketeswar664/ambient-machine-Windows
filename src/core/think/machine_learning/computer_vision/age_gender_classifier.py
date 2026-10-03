import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class AgeGenderClassifier:
    """
    Lightweight in-memory age and gender predictor for a single face image.

    Usage:
        classifier = AgeGenderClassifier(model_path="...")
        age, gender, confidence = classifier.predict(face_image_np)
    """
    def __init__(self, model_path: str, image_size: int = 224, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] AgeGenderClassifier using device: {self.device}")

        # Load TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Class labels
        self.class_names = [
            'adult_female', 'adult_male', 'elderly_female', 'elderly_male',
            'kid_female', 'kid_male', 'teen_female', 'teen_male'
        ]

    def predict(self, face_img: np.ndarray) -> tuple[str, str, float]:
        """
        Predict age and gender for a single face crop.

        Parameters:
        - face_img: numpy.ndarray in BGR color (as from OpenCV)

        Returns:
        - age_label: str (e.g. 'adult')
        - gender_label: str (e.g. 'female')
        - confidence: float (probability of the predicted class)
        """
        # Convert BGR numpy array to PIL RGB
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        # Preprocess and batch
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(tensor)
            # Handle tuple output
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, dim=1)

        cls_name = self.class_names[idx.item()]
        age, gender = cls_name.split('_', 1)
        return age, gender, float(conf.item())
