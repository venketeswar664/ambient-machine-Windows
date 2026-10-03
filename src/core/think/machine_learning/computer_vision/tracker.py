from collections import defaultdict, deque
import numpy as np
from scipy.spatial.distance import euclidean
from pykalman import KalmanFilter
import time

class Tracker:
    def __init__(self, window_size=20, max_distance=50, max_disappearance_time=60):
        self.window_size = window_size
        self.window_data = defaultdict(lambda: defaultdict(deque))
        self.labels = []
        self.masks = []
        self.boxes = []
        self.track_ids = []
        self.confidence = []
        self.trackers = {}
        self.max_distance = max_distance  # Maximum distance to consider for reassociation
        self.max_disappearance_time = max_disappearance_time  # Maximum disappearance time in seconds
        self.track_id_count = 0
        self.reassigned_track_ids = {}  # Stores reassigned track IDs and their original IDs
        self.last_seen = {}  # Tracks the last time a track ID was seen
        self.kalman_states = {}  # Stores Kalman filter states

    def update_track_id(self, labels, masks, boxes, track_ids, confidence):
        self.labels = labels
        self.boxes = boxes
        self.track_ids = track_ids
        self.confidence = confidence

        new_track_ids = []

        # Update the window data with the new values
        for i, track_id in enumerate(track_ids):
            current_time = time.time()

            if track_id in self.reassigned_track_ids:
                reassigned_track_id = self.reassigned_track_ids[track_id]
            else:
                reassigned_track_id = self.reassociate_track_id(track_id, boxes[i])
                if reassigned_track_id != track_id:
                    self.reassigned_track_ids[track_id] = reassigned_track_id

            new_track_ids.append(reassigned_track_id)
            self.last_seen[reassigned_track_id] = current_time

            self.window_data[reassigned_track_id]['labels'].append(labels[i])
            self.window_data[reassigned_track_id]['boxes'].append(boxes[i])
            self.window_data[reassigned_track_id]['confidence'].append(confidence[i])

            # Ensure the deque does not exceed the window size
            if len(self.window_data[reassigned_track_id]['labels']) > self.window_size:
                self.window_data[reassigned_track_id]['labels'].popleft()
                self.window_data[reassigned_track_id]['boxes'].popleft()
                self.window_data[reassigned_track_id]['confidence'].popleft()

            # Update Kalman Filter with the new detection
            self.update_kalman_filter(reassigned_track_id, boxes[i])

        # Remove track IDs that have been missing for more than the maximum disappearance time
        self.cleanup_old_track_ids()

        self.track_ids = new_track_ids
        return self.track_ids

    def update_labels(self):
        # Calculate the average labels and confidence scores
        calculated_labels = []

        for track_id in self.track_ids:
            labels_window = list(self.window_data[track_id]['labels'])

            # Calculate average label
            average_label = max(set(labels_window), key=labels_window.count)
            calculated_labels.append(average_label)

        return calculated_labels

    def reassociate_track_id(self, track_id, box):
        # Calculate the center of the bounding box
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        min_distance = float('inf')
        best_match = track_id

        # Reassociate track ID based on the closest previous track ID within the max distance
        for previous_track_id, data in self.window_data.items():
            previous_boxes = list(data['boxes'])
            if not previous_boxes:
                continue
            prev_center_x = (previous_boxes[-1][0] + previous_boxes[-1][2]) / 2
            prev_center_y = (previous_boxes[-1][1] + previous_boxes[-1][3]) / 2

            distance = euclidean((center_x, center_y), (prev_center_x, prev_center_y))
            if distance < self.max_distance and distance < min_distance:
                min_distance = distance
                best_match = previous_track_id

        # If no reassociation is possible, return the original track ID
        return best_match

    def update_kalman_filter(self, track_id, box):
        # Calculate the center of the bounding box
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        if track_id not in self.kalman_states:
            # Initialize the Kalman filter state for this track_id
            kf = KalmanFilter(
                transition_matrices=np.eye(4), 
                observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            )
            initial_state_mean = [center_x, center_y, 0, 0]
            initial_state_covariance = np.eye(4) * 1000  # Large initial uncertainty
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

    def cleanup_old_track_ids(self):
        current_time = time.time()
        to_delete = []

        for track_id, last_seen_time in self.last_seen.items():
            if current_time - last_seen_time > self.max_disappearance_time:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.window_data[track_id]
            del self.last_seen[track_id]
            if track_id in self.trackers:
                del self.trackers[track_id]
            if track_id in self.kalman_states:
                del self.kalman_states[track_id]
            # Remove reassigned track IDs that are no longer needed
            self.reassigned_track_ids = {k: v for k, v in self.reassigned_track_ids.items() if v != track_id}
