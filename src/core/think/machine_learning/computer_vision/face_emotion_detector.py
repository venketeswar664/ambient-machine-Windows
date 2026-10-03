from threading import Thread
import cv2
import torch
import numpy as np
import datetime
import os
from ultralytics import YOLO
from shapely.geometry import Point, Polygon, box
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from src.database.database import Database
from src.database.metadata_handler import MetadataHandler
from src.utils import plot
from src.core.actuators.video_writer import VideoWriterManager
from src.database.schemas.sentinel_poc_schema import Zone
from PIL import Image
import logging

class EmotionPredictor:
    def __init__(self, model_path='/home/sentinel/Projects/attendance-system/ambient-machine/src/models/attention_gimmefive_best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def load_model(self, model_path):
        model = GimmeFiveWithAttention(num_classes=7)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def predict(self, face_img):
        tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs, _, _ = self.model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        return {emotion: float(prob) for emotion, prob in zip(self.classes, probs)}

class EmotionPipeline:
    def __init__(self, model_path="/home/sentinel/Projects/attendance-system/ambient-machine/src/models/attention_gimmefive_best_model.pth"):
        self.emotion_predictor = EmotionPredictor(model_path) if os.path.exists(model_path) else None

    def process_face(self, face_image):
        try:
            if face_image.size == 0 or self.emotion_predictor is None:
                return "unknown"
            
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            emotion_probs = self.emotion_predictor.predict(rgb_face)
            return max(emotion_probs.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logging.error(f"Emotion detection error: {str(e)}")
            return "unknown"

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super().__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = torch.nn.Conv2d(in_features_l, attn_features, kernel_size=1, bias=False)
        self.W_g = torch.nn.Conv2d(in_features_g, attn_features, kernel_size=1, bias=False)
        self.phi = torch.nn.Conv2d(attn_features, 1, kernel_size=1, bias=True)
        
    def forward(self, l, g):
        N, C, H, W = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = torch.nn.functional.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(torch.nn.functional.relu(l_ + g_))
        
        if self.normalize_attn:
            a = torch.nn.functional.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)
        else:
            a = torch.sigmoid(c)
        f = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            output = f.view(N, C, -1).sum(dim=2)
        else:
            output = torch.nn.functional.adaptive_avg_pool2d(f, (1,1)).view(N, C)
        return a, output

class Detector:
    def __init__(self, config, logger=None, save_video=True, save_raw_video=False):
        print("Initialization taking place")
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize core parameters
        self.device = config.get('device', 'cuda:0')
        self.model_paths = config.get('model_paths', {})
        self.plot = config.get('plot', False)
        self.conf = config.get('conf', 0.6)
        self.class_ids = config.get('class_ids', [0])

        self.ip_camera = config.get('ip_camera')
        if not self.ip_camera:
            raise ValueError("IP Camera configuration is missing")
        self.fps = self.ip_camera.fps


        # Initialize video parameters
        date = datetime.datetime.now().date()
        self.save_raw_video = save_raw_video
        self.save_video = True
        self.video_writer = None
        if self.save_video:
            video_dir = os.path.join(f"./results/videos/FM/{date}", self.ip_camera.device_name)
            self.video_writer = VideoWriterManager(
                save_videos_flag=True,
                date=date,
                results_base_dir=video_dir,
                frame_size=(int(self.ip_camera.frame_width ), int(self.ip_camera.frame_height)),
                fps=self.fps/5,
                logger=self.logger
            )

        # Initialize models
        self._init_face_model()
        self._init_embedding_model()
        self._init_emotion_model()

        # Initialize camera connection
        
        # Initialize zones
        self.zones = self.ip_camera.zones
        self.zone_dict = self._init_zones()

        # Initialize metadata handler
        self.metadata_handler = MetadataHandler(
            self.ip_camera.ip_address,
            self.ip_camera.device_name,
            logger=self.logger
        )

        # Initialize video writers
        self._init_video_writers()

    def _init_face_model(self):
        """Initialize YOLO face detection model"""
        face_model_path = self.model_paths.get('face_model')
        if not face_model_path or not os.path.exists(face_model_path):
            raise FileNotFoundError(f"Face model not found at {face_model_path}")
        
        self.face_model = YOLO(face_model_path).to(self.device)
        self.logger.info(f"Loaded face model from {face_model_path}")

    def _init_emotion_model(self):
        """Initialize emotion recognition model"""
        emotion_model_path = self.model_paths.get('emotion_model')
        if emotion_model_path and os.path.exists(emotion_model_path):
            self.emotion_model = EmotionPipeline(emotion_model_path)
            self.logger.info(f"Loaded emotion model from {emotion_model_path}")
        else:
            self.logger.warning("Emotion model path not found or invalid")
            self.emotion_model = None

    def _init_embedding_model(self):
        """Initialize face embedding model"""
        self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embedding_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _init_zones(self):
        """Initialize zone data from database"""
        zone_dict = {}
        try:
            if not isinstance(self.zones, (list, set)):
                self.zones = [self.zones]

            for zone in Zone.objects(id__in=self.zones):
                if hasattr(zone, 'roi') and zone.roi:
                    # Convert ROI to list of (x, y) tuples
                    vertices = []
                    for point in np.array(zone.roi).reshape(-1, 2):
                        x, y = map(int, point)
                        vertices.append((x, y))
                    
                    if len(vertices) >= 3:
                        zone_dict[zone.name] = Polygon(vertices)
                        self.logger.debug(f"Added zone: {zone.name} with {len(vertices)} points")
                    else:
                        self.logger.warning(f"Invalid ROI for zone {zone.name}")
        except Exception as e:
            self.logger.error(f"Zone initialization error: {str(e)}")
        
        return zone_dict

    def _init_video_writers(self):
        """Initialize video writer components"""
        date = datetime.datetime.now().date()
        frame_size = (int(self.ip_camera.frame_width), int(self.ip_camera.frame_height))

        if self.save_video:
            video_dir = os.path.join(f"./results/videos/FM/{date}", self.ip_camera.device_name)
            self.video_writer = VideoWriterManager(
                save_videos_flag=True,
                date=date,
                results_base_dir=video_dir,
                frame_size=frame_size,
                fps=self.ip_camera.fps // 5,
                logger=self.logger
            )

        if self.save_raw_video:
            raw_video_dir = os.path.join(f"./results/raw_videos/FM/{date}", self.ip_camera.device_name)
            self.raw_video_writer = VideoWriterManager(
                save_videos_flag=True,
                date=date,
                results_base_dir=raw_video_dir,
                frame_size=frame_size,
                fps=self.ip_camera.fps,
                logger=self.logger
            )

    # def _is_in_zone(self, bbox):
    #     """Check if bounding box is in any defined zone"""
    #     x1, y1, x2, y2 = map(int, bbox)
    #     bbox_poly = box(x1, y1, x2, y2)
    #     # print("check the zone ")
        
    #     for zone_name, zone_poly in self.zone_dict.items():
    #         try:
    #             iou = bbox_poly.intersection(zone_poly).area / bbox_poly.union(zone_poly).area
    #             if iou > 0.001:
    #                 # print("in zone")
    #                 return True
    #         except Exception as e:
    #             self.logger.warning(f"Zone check error for {zone_name}: {str(e)}")
        
    #     return False

    def _is_in_zone(self, bbox):
        """Check if all four corners of the bounding box are inside any defined zone"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Create points for each corner of the bounding box
        top_left = Point(x1, y1)
        top_right = Point(x2, y1)
        bottom_left = Point(x1, y2)
        bottom_right = Point(x2, y2)
        corners = [top_left, top_right, bottom_left, bottom_right]
        
        for zone_name, zone_poly in self.zone_dict.items():
            try:
                # Check if all corners are inside the zone
                if all(zone_poly.contains(corner) for corner in corners):
                    return True
            except Exception as e:
                self.logger.warning(f"Zone check error for {zone_name}: {str(e)}")
        
        return False

    def _extract_face_data(self, frame, bbox):
        """Process individual face detection"""
        x1, y1, x2, y2 = map(int, bbox)
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None

        try:
            # Convert to PIL Image if necessary
            if isinstance(face_img, np.ndarray):
                face_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Get embedding
            face_tensor = self.embedding_transform(face_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedding_model(face_tensor).cpu().numpy().tolist()

            # Get emotion
            emotion = "unknown"
            if self.emotion_model:
                emotion = self.emotion_model.process_face(cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
                

            return {
                "bbox": [x1, y1, x2, y2],
                "embedding": embedding,
                "emotion": emotion,
                "confidence": float(box.conf) if hasattr(box, 'conf') else 0.8
            }
        except Exception as e:
            self.logger.error(f"Face processing error: {str(e)}")
            return None

    # def process_frame(self, frame):
    #     """Main processing pipeline for a single frame"""
    #     results = {
    #         "labels": [],
    #         "bboxes": [],
    #         "confidences": [],
    #         "emotions": [],
    #         "frame": frame
    #     }

    #     try:
    #         if self.zone_dict:
    #             results['frame'] = plot.plot_shapes(results['frame'], self.zone_dict)

    #         detections = self.face_model.predict(frame, conf=self.conf, verbose=False)[0]
            
    #         for box in detections.boxes:
    #             bbox = box.xyxy[0].cpu().numpy()
    #             if not self._is_in_zone(bbox):
    #                 continue

    #             face_data = self._extract_face_data(frame, bbox)
    #             if not face_data:
    #                 continue

    #             results['labels'].append("recognized_person")
    #             results['bboxes'].append(face_data['bbox'])
    #             results['confidences'].append(face_data['confidence'])
    #             results['emotions'].append(face_data['emotion'])

    #             if self.plot:
    #                 plot.plot_faces_with_labels(
    #                     frame, face_data['bbox'],
    #                     f"{face_data['emotion']} ({face_data['confidence']:.2f})",
    #                     (0, 255, 0)
    #                 )

            
    #     except Exception as e:
    #         self.logger.error(f"Frame processing error: {str(e)}")

    #     if self.save_video:
    #             self.video_writer.write_frame(results['frame'], datetime.datetime.now())
    #             # print("frame saved processed")
            
    #         # Prepare metadata with native Python types
    #         # ... [rest of metadata handling code] ...

    #     if self.save_raw_video:
    #             self.raw_video_writer.write_frame(frame, datetime.datetime.now())
    #             # print("frame saved ")

    #     return results

    def process_frame(self, frame):
        """Main processing pipeline for a single frame"""
        results = {
            "labels": [],
            "bboxes": [],
            "confidences": [],
            "emotions": [],
            "frame": frame.copy()  # Create a copy to avoid modifying the original
        }

        try:
            # Always plot zones regardless of self.plot flag
            if self.zone_dict:
                # Plot zones with semi-transparent fill
                frame_with_zones = frame.copy()
                for zone_name, zone_poly in self.zone_dict.items():
                    # Get zone vertices
                    if hasattr(zone_poly, 'exterior'):
                        # For Shapely 2.0+
                        vertices = np.array(zone_poly.exterior.coords).astype(np.int32)
                    else:
                        # For older Shapely versions
                        vertices = np.array(zone_poly.exterior.coords).astype(np.int32)
                    
                    # Create a mask for this zone
                    mask = np.zeros_like(frame)
                    cv2.fillPoly(mask, [vertices], (0, 128, 0))  # Green semi-transparent fill
                    
                    # Apply mask with transparency
                    alpha = 0.3  # Transparency factor
                    frame_with_zones = cv2.addWeighted(frame_with_zones, 1.0, mask, alpha, 0)
                    
                    # Draw zone outline
                    cv2.polylines(frame_with_zones, [vertices], True, (0, 255, 0), 2)
                    
                    # Add zone name
                    centroid = np.mean(vertices, axis=0).astype(int)
                    cv2.putText(frame_with_zones, zone_name, tuple(centroid), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                results['frame'] = frame_with_zones

            detections = self.face_model.predict(frame, conf=self.conf, verbose=False)[0]
            
            for box in detections.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                
                # Check if the face is in zone before processing
                if not self._is_in_zone(bbox):
                    # Optionally, draw red boxes for faces outside zones
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(results['frame'], (x1, y1), (x2, y2), (0, 0, 255), 2)
                    continue

                face_data = self._extract_face_data(frame, bbox)
                if not face_data:
                    continue

                results['labels'].append("recognized_person")
                results['bboxes'].append(face_data['bbox'])
                results['confidences'].append(face_data['confidence'])
                results['emotions'].append(face_data['emotion'])

                # Always plot detected faces that are in zone
                x1, y1, x2, y2 = map(int, face_data['bbox'])
                cv2.rectangle(results['frame'], (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add emotion label
                label = f"{face_data['emotion']} ({face_data['confidence']:.2f})"
                cv2.putText(results['frame'], label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")

        # Always save the processed frame if enabled
        if self.save_video:
            self.video_writer.write_frame(results['frame'], datetime.datetime.now())
            
        if self.save_raw_video:
            self.raw_video_writer.write_frame(frame, datetime.datetime.now())

        return results

    def process(self):
        """Main processing loop"""
        print("start")
        try:
            while self.ip_camera.is_open:
                ret, frame = self.ip_camera.capture.read()
                if not ret:
                    break

                results = self.process_frame(frame)

                # Save processed frame (with annotations) if enabled
                
                

                # Prepare metadata with native Python types
                # metadata = {
                #     "timestamp": datetime.datetime.now().isoformat(),
                #     "detections": {
                #         "labels": [str(label) for label in results['labels']],
                #         "bboxes": [list(map(int, bbox)) for bbox in results['bboxes']],
                #         "confidences": [float(conf) for conf in results['confidences']],
                #         "emotions": [str(emotion) for emotion in results['emotions']]
                #     }
                # }

                # self.metadata_handler.process(metadata)

                # if self.save_raw_video:
                #     self.raw_video_writer.write_frame(frame, datetime.datetime.now())

        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        finally:
            print("face detection pipeline complete ")
            self._cleanup()

    def _cleanup(self):
        """Release resources"""
        try:
            self.ip_camera.capture.release()
            if self.video_writer:
                self.video_writer.end_current_video()
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

class GimmeFiveWithAttention(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,3,padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), torch.nn.Dropout(0.2))
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), torch.nn.Dropout(0.2))
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128,256,3,padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), torch.nn.Dropout(0.2))
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(256,512,3,padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), torch.nn.Dropout(0.2))
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(512,1024,3,padding=1), torch.nn.BatchNorm2d(1024), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2))
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.attn1 = AttentionBlock(256, 1024, 256, up_factor=4)
        self.attn2 = AttentionBlock(512, 1024, 256, up_factor=2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024+256+512, 2048), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(2048,1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes))

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        g = self.adaptive_pool(x5).flatten(1)
        _, g1 = self.attn1(x3, x5)
        _, g2 = self.attn2(x4, x5)
        g_hat = torch.cat([g, g1, g2], dim=1)
        return self.classifier(g_hat), None, None