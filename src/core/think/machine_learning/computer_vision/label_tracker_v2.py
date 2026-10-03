# -*- coding: utf-8 -*-
'''
Created on 2025-03-07 06:16
Project: ambient-machine
@author: Aryan Sinha
@email: aryan.sinha@neophyte.ai
'''



from collections import defaultdict, deque
import time
from statistics import mode, StatisticsError

class ImprovedLabelTracker:
    def __init__(self, time_limit=10, window_size=10):
        """
        Args:
            time_limit (int): Time in seconds for retaining label history for each track ID.
            window_size (int): Maximum number of recent labels to consider for smoothing.
        """
        self.time_limit = time_limit
        self.window_size = window_size
        # For each track ID, store a deque for recent labels, the current stable label, and the last update time.
        self.window_data = defaultdict(lambda: {
            "label_deque": deque(maxlen=self.window_size),
            "label": None,
            "update_time": 0
        })

    def update_labels(self, track_ids, labels):
        current_time = time.time()
        for track_id, label in zip(track_ids, labels):
            data = self.window_data[track_id]
            data["label_deque"].append(label)
            data["update_time"] = current_time
            # Compute the mode for the current window of labels.
            try:
                data["label"] = mode(list(data["label_deque"]))
            except StatisticsError:
                data["label"] = label
        self._check_time_limit(track_ids)

    def _check_time_limit(self, current_track_ids):
        current_time = time.time()
        # Remove track IDs that haven't been updated within the time limit.
        outdated = [
            tid for tid, data in self.window_data.items() 
            if tid not in current_track_ids and (current_time - data["update_time"] > self.time_limit)
        ]
        for tid in outdated:
            del self.window_data[tid]

    def get_label(self, track_ids):
        return [self.window_data[tid]["label"] if tid in self.window_data else None for tid in track_ids]
