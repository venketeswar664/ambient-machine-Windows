import tempfile
import cv2
import os
import datetime
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from ultralytics import YOLO
from mongoengine import Document, DateTimeField, StringField
from pymongo.errors import DuplicateKeyError
from src.utils.thumbnail_manager import AzureDatedImageUploader
from src.utils.embeddings_extractor import ResNetAttentionEmbedder
from src.database.schemas.sentinel_poc_schema import Stores
# from age_gender_classifier import AgeGenderClassifier
# from face_detection import predict_and_crop_faces

# Create debug directory if it doesn't exist
DEBUG_DIR = "debug_face_detection"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Load YOLO model for face detection
model = YOLO("/home/sentinel/projects/ambient-machine/src/models/yolov11m-face.pt")

def predict_and_crop_faces(img, conf=0.5, padding_ratio=0.2, debug_info=None):
    """
    Detect faces using YOLO and return bounding boxes and cropped face regions.
    Enhanced with debug functionality.

    :param img: Input image (numpy array)
    :param conf: Confidence threshold for detection
    :param padding_ratio: Padding ratio to expand bounding boxes
    :param debug_info: Dict with debug info like track_id, doc_id for saving debug images
    :return: dict with 'boxes' and 'crops'
    """
    results = model.predict(img, conf=conf)
    boxes = []
    crops = []
    
    # Debug: Save original image with detection results
    debug_img = img.copy()
    detection_count = 0

    for result in results:
        if result.boxes is not None:
            detection_count += len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Draw detection box on debug image
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_img, f'Face: {confidence:.2f}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Add padding
                face_w = x2 - x1
                face_h = y2 - y1
                pad_w = int(face_w * padding_ratio)
                pad_h = int(face_h * padding_ratio)

                x1_padded = max(0, x1 - pad_w)
                y1_padded = max(0, y1 - pad_h)
                x2_padded = min(img.shape[1], x2 + pad_w)
                y2_padded = min(img.shape[0], y2 + pad_h)

                boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])
                crop = img[y1_padded:y2_padded, x1_padded:x2_padded]
                if crop.size > 0:
                    crops.append(crop)
                    
                    # Draw padded box on debug image
                    cv2.rectangle(debug_img, (x1_padded, y1_padded), (x2_padded, y2_padded), (255, 0, 0), 1)

    # Save debug images
    if debug_info:
        track_id = debug_info.get('track_id', 'unknown')
        doc_id = debug_info.get('doc_id', 'unknown')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if detection_count == 0:
            # No faces detected - save original image for analysis
            debug_filename = f"{DEBUG_DIR}/no_face_{track_id}_{doc_id}_{timestamp}.jpg"
            cv2.imwrite(debug_filename, img)
            print(f"[DEBUG] No faces detected - saved original image: {debug_filename}")
            
            # Also save image info
            info_filename = f"{DEBUG_DIR}/no_face_{track_id}_{doc_id}_{timestamp}_info.txt"
            with open(info_filename, 'w') as f:
                f.write(f"Track ID: {track_id}\n")
                f.write(f"Doc ID: {doc_id}\n")
                f.write(f"Image shape: {img.shape}\n")
                f.write(f"Confidence threshold: {conf}\n")
                f.write(f"Detections found: {detection_count}\n")
                f.write(f"Timestamp: {timestamp}\n")
        else:
            # Faces detected - save image with bounding boxes
            debug_filename = f"{DEBUG_DIR}/detected_{track_id}_{doc_id}_{timestamp}_{len(crops)}faces.jpg"
            cv2.imwrite(debug_filename, debug_img)
            print(f"[DEBUG] {len(crops)} faces detected - saved annotated image: {debug_filename}")
            
            # Save individual face crops
            for i, crop in enumerate(crops):
                crop_filename = f"{DEBUG_DIR}/crop_{track_id}_{doc_id}_{timestamp}_face{i}.jpg"
                cv2.imwrite(crop_filename, crop)

    return {
        "boxes": boxes,
        "crops": crops,
        "detection_count": detection_count
    }


class HumanProfile(Document):
    track_id = StringField(required=True)
    store_id = StringField(required=True)
    time_stamp = DateTimeField(required=True)
    last_updated_time = DateTimeField(required=True)
    age = StringField()
    gender = StringField()
    embedding = StringField(required=True)
    thumbnail = StringField(required=True)

    def __str__(self):
        return f"HumanProfile(track_id={self.track_id}, age={self.age})"

    meta = {
        'collection': 'human_profile'
    }
class AgeGenderClassifier:
    def __init__(self, model_path, image_size: int = 224, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")
        self.model = self._load_model(model_path)
        self.image_size = image_size
        self.class_names = [
            'adult_female', 'adult_male', 'elderly_female', 'elderly_male',
            'kid_female', 'kid_male', 'teen_female', 'teen_male'
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path: str):
        print("[INFO] Loading TorchScript model...")
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def infer_single_image(self, image_input):
        """
        Accepts:
          - file path (str)
          - PIL.Image
          - NumPy array (OpenCV BGR)
        Returns a dict with keys: age, gender, confidence, inference_time
        """
        # --- load/convert to PIL ---
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError("Unsupported input type. Use file path, PIL.Image, or OpenCV image.")

        # --- preprocess ---
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # --- inference ---
        with torch.no_grad():
            start = time.time()
            output = self.model(tensor)
            elapsed = time.time() - start

        if isinstance(output, tuple):
            output = output[0]

        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

        label = self.class_names[pred.item()]
        age, gender = label.split("_")
        return {
            "age": age,
            "gender": gender,
            "confidence": conf.item(),
            "inference_time": elapsed
        }

class EntryExitHandler:
    """
    Handles entry events by selecting the best YOLO-confidence crop, then:
      1) Detect face in that crop for age/gender prediction
      2) Predict age/gender on the face region
      3) Upload the full-body crop to Azure Blob Storage
      4) Extract embeddings from the full-body crop
      5) Store results in MongoDB
    """

    def __init__(self):
        self.azure_uploader = AzureDatedImageUploader()
        self.age_gender_classifier = AgeGenderClassifier(model_path="/home/sentinel/projects/ambient-machine/src/models/age_gender_face.pt")
        # self.embedder = ResNetAttentionEmbedder(model_path="/home/sentinel/Projects/ambient-machine/src/models/embedding_extractor.pth")
        # self.embedding_handler = MilvusEmbeddingHandler()

    def process(self, track_id: int, crops_list: list, confidences: list, label_name: str):
        """
        Main processing function to handle the full flow:
        - Detect faces
        - Predict age/gender
        - Upload full-body crop to Azure
        - Store information in MongoDB
        """

        # 1) Choose best crop by YOLO confidence
        best_idx = int(np.argmax(confidences))
        best_crop = crops_list[best_idx]

        # 2) Detect faces and get cropped face regions
        face_detection_results = predict_and_crop_faces(best_crop, conf=0.5, padding_ratio=0.2)
        face_boxes = face_detection_results["boxes"]
        face_crops = face_detection_results["crops"]

        # 3) Process the first detected face (You can adjust this if you want to handle multiple faces)
        age = "unknown"
        gender = "unknown"
        if len(face_crops) > 0:
            face_crop = face_crops[0]
            face_age_gender = self.age_gender_classifier.infer_single_image(face_crop)

            # Extract age and gender from the classifier output
            age = face_age_gender["age"]
            gender = face_age_gender["gender"]

        # 4) Upload full-body crop to Azure Blob Storage
        ts = datetime.datetime.now(datetime.timezone.utc)
        image_name = f"{track_id}.jpg"
        try:
            thumbnail_url = self.azure_uploader.upload_image(
                best_crop,
                image_name=image_name,
                label=str(label_name),
                timestamp=ts
            )
        except Exception as e:
            print(f"Error uploading image: {e}")
            thumbnail_url = None

        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(tmp.name, best_crop)
        tmp.close()

        # Generate embeddings (optional, if you want to store in DB)
        # embedding = self.embedder.extract_embeddings(tmp.name)  # Uncomment if embeddings are required

        # 5) Store/Update in MongoDB
        existing_profile = HumanProfile.objects(track_id=str(track_id)).first()

        if existing_profile:
            print(f"Track ID {track_id} already exists in the database.")
            # Update the existing profile with age, gender, and other information
            HumanProfile.objects(track_id=track_id).update_one(
                set__age=age,
                set__gender=gender,
                set__thumbnail=thumbnail_url,
                set__last_updated_time=datetime.datetime.now(datetime.timezone.utc)
            )
        else:
            print(f"Track ID {track_id} does not exist in the database.")
            # Create a new profile record
            record = HumanProfile(
                track_id=str(track_id),
                store_id=str("680c6de754206adc15d9ab4f"),
                time_stamp=ts,
                last_updated_time=ts,
                age=age,
                gender=gender,
                embedding="embedding_placeholder",  # Optional: add embedding if needed
                thumbnail=thumbnail_url
            )
            try:
                record.save()
                print(f"Saved: {record}")
            except DuplicateKeyError:
                # Update existing record if duplicate key error occurs
                HumanProfile.objects(track_id=track_id).update_one(
                    set__age=age,
                    set__gender=gender,
                    set__thumbnail=thumbnail_url,
                    set__last_updated_time=datetime.datetime.now(datetime.timezone.utc)
                )

        # Optional: Log to file for debugging or auditing purposes
        # Log here if necessary


# class AgeGenderClassifier:
#     def __init__(self, age_model_path, gender_model_path, image_size: int = 224, device=None):
#         self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"[INFO] Using device: {self.device}")
        
#         # Load separate models for age and gender prediction
#         self.age_model = self._load_model(age_model_path)
#         self.gender_model = self._load_model(gender_model_path)

#         self.image_size = image_size
#         self.age_class_names = [
#             'elderly', 'adult', 'kid', 'teen'
#         ]
#         self.gender_class_names = [
#             'female', 'male'
#         ]
        
#         self.transform = transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])

#     def _load_model(self, model_path: str):
#         """
#         Loads a TorchScript model.
#         """
#         print(f"[INFO] Loading TorchScript model from {model_path}...")
#         model = torch.jit.load(model_path, map_location=self.device)
#         model.eval()
#         return model

#     def infer_single_image(self, image_input):
#         """
#         Accepts:
#           - file path (str)
#           - PIL.Image
#           - NumPy array (OpenCV BGR)
#         Returns a dict with keys: age, gender, confidence, inference_time
#         """
#         # --- load/convert to PIL ---
#         if isinstance(image_input, str):
#             image = Image.open(image_input).convert('RGB')
#         elif isinstance(image_input, np.ndarray):
#             image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
#         elif isinstance(image_input, Image.Image):
#             image = image_input
#         else:
#             raise TypeError("Unsupported input type. Use file path, PIL.Image, or OpenCV image.")

#         # --- preprocess ---
#         tensor = self.transform(image).unsqueeze(0).to(self.device)

#         # --- inference for age ---
#         with torch.no_grad():
#             start = time.time()
#             age_output = self.age_model(tensor)
#             elapsed = time.time() - start

#         # --- inference for gender ---
#         with torch.no_grad():
#             start = time.time()
#             gender_output = self.gender_model(tensor)
#             gender_elapsed = time.time() - start

#         # Process outputs
#         age_probs = F.softmax(age_output, dim=1)
#         gender_probs = F.softmax(gender_output, dim=1)

#         # Get the most probable age and gender
#         age_conf, age_pred = torch.max(age_probs, dim=1)
#         gender_conf, gender_pred = torch.max(gender_probs, dim=1)

#         age_label = self.age_class_names[age_pred.item()]
#         gender_label = self.gender_class_names[gender_pred.item()]

#         return {
#             "age": age_label,
#             "gender": gender_label,
#             "age_confidence": age_conf.item(),
#             "gender_confidence": gender_conf.item(),
#             "age_inference_time": elapsed,
#             "gender_inference_time": gender_elapsed
#         }

# class EntryExitHandler:
#     """
#     Handles entry events by selecting the best YOLO-confidence crop, then:
#       1) Detect face in that crop for age/gender prediction
#       2) Predict age/gender on the face region
#       3) Upload the full-body crop to Azure Blob Storage
#       4) Extract embeddings from the full-body crop
#       5) Store results in MongoDB
#     """

#     def __init__(self):
#         self.azure_uploader = AzureDatedImageUploader()
#         # Pass separate model paths for age and gender
#         self.age_gender_classifier = AgeGenderClassifier(
#             age_model_path="/home/sentinel/projects/ambient-machine/src/models/age_classify_yolo.pt",
#             gender_model_path="/home/sentinel/projects/ambient-machine/src/models/gender_classify_yolo.pt"
#         )
#         # self.embedder = ResNetAttentionEmbedder(model_path="/home/sentinel/Projects/ambient-machine/src/models/embedding_extractor.pth")
#         # self.embedding_handler = MilvusEmbeddingHandler()

#     def process(self, track_id: int, crops_list: list, confidences: list, label_name: str):
#         """
#         Main processing function to handle the full flow:
#         - Detect faces
#         - Predict age/gender
#         - Upload full-body crop to Azure
#         - Store information in MongoDB
#         """

#         # 1) Choose best crop by YOLO confidence
#         best_idx = int(np.argmax(confidences))
#         best_crop = crops_list[best_idx]

#         # 2) Detect faces and get cropped face regions
#         face_detection_results = predict_and_crop_faces(best_crop, conf=0.5, padding_ratio=0.2)
#         face_boxes = face_detection_results["boxes"]
#         face_crops = face_detection_results["crops"]

#         # 3) Process the first detected face (You can adjust this if you want to handle multiple faces)
#         age = "unknown"
#         gender = "unknown"
#         if len(face_crops) > 0:
#             face_crop = face_crops[0]
#             face_age_gender = self.age_gender_classifier.infer_single_image(face_crop)

#             # Extract age and gender from the classifier output
#             age = face_age_gender["age"]
#             gender = face_age_gender["gender"]

#         # 4) Upload full-body crop to Azure Blob Storage
#         ts = datetime.datetime.now(datetime.timezone.utc)
#         image_name = f"{track_id}.jpg"
#         try:
#             thumbnail_url = self.azure_uploader.upload_image(
#                 best_crop,
#                 image_name=image_name,
#                 label=str(label_name),
#                 timestamp=ts
#             )
#         except Exception as e:
#             print(f"Error uploading image: {e}")
#             thumbnail_url = None

#         tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
#         cv2.imwrite(tmp.name, best_crop)
#         tmp.close()

#         # Generate embeddings (optional, if you want to store in DB)
#         # embedding = self.embedder.extract_embeddings(tmp.name)  # Uncomment if embeddings are required

#         # 5) Store/Update in MongoDB
#         existing_profile = HumanProfile.objects(track_id=str(track_id)).first()

#         if existing_profile:
#             print(f"Track ID {track_id} already exists in the database.")
#             # Update the existing profile with age, gender, and other information
#             HumanProfile.objects(track_id=track_id).update_one(
#                 set__age=age,
#                 set__gender=gender,
#                 set__thumbnail=thumbnail_url,
#                 set__last_updated_time=datetime.datetime.now(datetime.timezone.utc)
#             )
#         else:
#             print(f"Track ID {track_id} does not exist in the database.")
#             # Create a new profile record
#             record = HumanProfile(
#                 track_id=str(track_id),
#                 time_stamp=ts,
#                 last_updated_time=ts,
#                 age=age,
#                 gender=gender,
#                 embedding="embedding_placeholder",  # Optional: add embedding if needed
#                 thumbnail=thumbnail_url
#             )
#             try:
#                 record.save()
#                 print(f"Saved: {record}")
#             except DuplicateKeyError:
#                 # Update existing record if duplicate key error occurs
#                 HumanProfile.objects(track_id=track_id).update_one(
#                     set__age=age,
#                     set__gender=gender,
#                     set__thumbnail=thumbnail_url,
#                     set__last_updated_time=datetime.datetime.now(datetime.timezone.utc)
#                 )

#         # Optional: Log to file for debugging or auditing purposes
#         # Log here if necessary
