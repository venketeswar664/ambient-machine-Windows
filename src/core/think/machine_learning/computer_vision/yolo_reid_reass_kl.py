import cv2
import torch
import numpy as np
import torchreid
from ultralytics import YOLO
from collections import defaultdict, deque
from scipy.spatial.distance import cosine, euclidean
from pykalman import KalmanFilter
import time

class YOLOReIDTracker:
    def __init__(self, yolo_model_path, conf_threshold=0.5, max_distance=50):
        self.yolo_model = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.reid_extractor = torchreid.utils.FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='',  # Use default pretrained weights
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.track_features = defaultdict(lambda: deque(maxlen=20))
        self.all_track_ids = {}  # Dictionary to keep track of all track IDs
        self.kalman_states = {}  # Stores Kalman filter states
        self.max_distance = max_distance  # Maximum distance to consider for reassociation

    def find_pose(self, frame):
        results = self.yolo_model.track(frame, persist=True, verbose=False, conf=self.conf_threshold, device=0)
        masks, boxes, track_ids, labels, confidence = None, None, None, None, None

        if len(results[0]) != 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            labels = results[0].boxes.cls.int().cpu().tolist()
            confidence = results[0].boxes.conf.cpu().tolist()

        frame = results[0].plot(conf=False, labels=False, boxes=False, masks=False)

        return labels, masks, boxes, track_ids, confidence, frame

    def extract_features(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 256))  # Resize to (256, 128) as expected by the model
        crop = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float()  # Convert to tensor
        features = self.reid_extractor(crop)
        return features.flatten()  # Flatten the features to 1-D

    def update_track_ids(self, frame, boxes, current_track_ids):
        current_features = [self.extract_features(frame, box) for box in boxes]
        updated_track_ids = []

        for i, features in enumerate(current_features):
            best_match_id = None
            best_match_score = float('inf')

            # Check against stored features
            for track_id, feature_deque in self.track_features.items():
                avg_feature = torch.mean(torch.stack(list(feature_deque)), dim=0)
                score = cosine(features.cpu().numpy(), avg_feature.cpu().numpy().flatten())

                if score < best_match_score:
                    best_match_score = score
                    best_match_id = track_id

            if best_match_score < 0.5:  # Adjust threshold as needed
                updated_track_ids.append(best_match_id)
                self.track_features[best_match_id].append(features)
            else:
                # Check against all historical track IDs
                for track_id, feature_deque in self.all_track_ids.items():
                    avg_feature = torch.mean(torch.stack(list(feature_deque)), dim=0)
                    score = cosine(features.cpu().numpy(), avg_feature.cpu().numpy().flatten())

                    if score < best_match_score:
                        best_match_score = score
                        best_match_id = track_id

                if best_match_score < 0.5:
                    updated_track_ids.append(best_match_id)
                    self.track_features[best_match_id].append(features)
                else:
                    new_track_id = max(self.all_track_ids.keys(), default=0) + 1
                    updated_track_ids.append(new_track_id)
                    self.track_features[new_track_id].append(features)
                    self.all_track_ids[new_track_id] = self.track_features[new_track_id]

        return updated_track_ids

    def reassociate_track_id(self, track_id, box):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        min_distance = float('inf')
        best_match = track_id

        for previous_track_id, data in self.track_features.items():
            if previous_track_id not in self.kalman_states:
                continue

            state_mean, _ = self.kalman_states[previous_track_id]
            pred_center_x, pred_center_y = state_mean[0], state_mean[1]

            distance = euclidean((center_x, center_y), (pred_center_x, pred_center_y))
            if distance < self.max_distance and distance < min_distance:
                min_distance = distance
                best_match = previous_track_id

        return best_match

    def update_kalman_filter(self, track_id, box):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        if track_id not in self.kalman_states:
            kf = KalmanFilter(
                transition_matrices=np.eye(4),
                observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            )
            initial_state_mean = [center_x, center_y, 0, 0]
            initial_state_covariance = np.eye(4) * 1000
            self.kalman_states[track_id] = (initial_state_mean, initial_state_covariance)
        else:
            kf = KalmanFilter(
                transition_matrices=np.eye(4),
                observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            )

        state_mean, state_covariance = self.kalman_states[track_id]
        state_mean, state_covariance = kf.filter_update(
            filtered_state_mean=state_mean,
            filtered_state_covariance=state_covariance,
            observation=[center_x, center_y]
        )
        self.kalman_states[track_id] = (state_mean, state_covariance)

    def predict_kalman(self, track_id):
        if track_id in self.kalman_states:
            state_mean, state_covariance = self.kalman_states[track_id]
            kf = KalmanFilter(
                transition_matrices=np.eye(4),
                observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            )
            state_mean, state_covariance = kf.filter_update(
                filtered_state_mean=state_mean,
                filtered_state_covariance=state_covariance,
                observation=None
            )
            self.kalman_states[track_id] = (state_mean, state_covariance)
            return state_mean
        return None

    def ensemble_update_track_ids(self, frame, boxes, current_track_ids):
        current_features = [self.extract_features(frame, box) for box in boxes]
        updated_track_ids = []
        assigned_track_ids = set()

        for i, features in enumerate(current_features):
            best_match_id = None
            best_match_score = float('inf')
            reassigned_track_id = None
            kalman_prediction_score = float('inf')

            # Check against stored features (ReID)
            for track_id, feature_deque in self.track_features.items():
                avg_feature = torch.mean(torch.stack(list(feature_deque)), dim=0)
                score = cosine(features.cpu().numpy(), avg_feature.cpu().numpy().flatten())

                if score < best_match_score:
                    best_match_score = score
                    best_match_id = track_id

            if best_match_score < 0.5 and best_match_id not in assigned_track_ids:  # Adjust threshold as needed
                reassigned_track_id = best_match_id
                assigned_track_ids.add(best_match_id)
                self.track_features[best_match_id].append(features)
            else:
                reassigned_track_id = self.reassociate_track_id(None, boxes[i])
                if reassigned_track_id not in assigned_track_ids:
                    assigned_track_ids.add(reassigned_track_id)
                    self.track_features[reassigned_track_id].append(features)
                else:
                    new_track_id = max(self.all_track_ids.keys(), default=0) + 1
                    reassigned_track_id = new_track_id
                    assigned_track_ids.add(new_track_id)
                    self.track_features[new_track_id].append(features)
                    self.all_track_ids[new_track_id] = self.track_features[new_track_id]

            self.update_kalman_filter(reassigned_track_id, boxes[i])
            updated_track_ids.append(reassigned_track_id)

        return updated_track_ids

    def plot_tracking_results(self, frame, boxes, track_ids, updated_track_ids):
        for i, (track_id, updated_track_id, box) in enumerate(zip(track_ids, updated_track_ids, boxes)):
            color = (0, 255, 0) if track_id == updated_track_id else (255, 0, 0)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{track_id}->{updated_track_id}', (x1, y1 +50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Tracking', frame)

def main(video_path, yolo_model_path):
    cap = cv2.VideoCapture(video_path)
    tracker = YOLOReIDTracker(yolo_model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        labels, masks, boxes, track_ids, confidence, processed_frame = tracker.find_pose(frame)

        if boxes and track_ids:
            updated_track_ids = tracker.ensemble_update_track_ids(frame, boxes, track_ids)
            tracker.plot_tracking_results(processed_frame, boxes, track_ids, updated_track_ids)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = '/home/pranjal/Project/sentinel/digitel/ambient-machine/data/videos/sentinel/results_3/camera-03_20240607_144202_8fps.mp4'
    yolo_model_path = '/home/pranjal/Project/sentinel/digitel/ambient-machine/src/models/Augmented_Digital_30Mar_2024.engine'
    main(video_path, yolo_model_path)
