# -*- coding: utf-8 -*-
'''
Created on 2025-03-07 06:20
Project: ambient-machine
@author: Aryan Sinha
@email: aryan.sinha@neophyte.ai
'''



import time
import copy
from collections import defaultdict, deque, Counter
from statistics import mode
from shapely.geometry import Polygon, Point

class ImprovedZoneInteractionTracker:
    def __init__(self, polygon_list, window=30, min_presence_time=2.0):
        """
        Args:
            polygon_list (list): List of polygons (each a list of (x, y) tuples) defining zones.
            window (int): Number of recent updates to smooth zone counts.
            min_presence_time (float): Minimum seconds a track ID must be continuously present in a zone to be counted.
        """
        self.polygons = [Polygon(p) for p in polygon_list]
        self.image_width = None
        self.image_height = None
        # For each zone (polygon index), we store a dict mapping track_id -> entry_time.
        self.zone_interaction_dict = defaultdict(lambda: {
            'track_info': dict(),  # track_id -> entry_time
            'most_frequent_track_ids': [],
            'most_frequent_count': 0
        })
        self.window_size = window
        # For smoothing, maintain a deque of counts and track ID lists for each zone.
        self.window_data = defaultdict(lambda: {
            "count": deque(maxlen=self.window_size),
            "track_ids_list": deque(maxlen=self.window_size)
        })
        self.min_presence_time = min_presence_time

    def set_image_size(self, image_size):
        self.image_width, self.image_height = image_size

    def update(self, boxes, track_ids, labels):
        current_time = time.time()
        # Update each zone's track info based on current detections.
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            current_position = self._get_center(bbox)
            point = Point(current_position)
            # Process every polygon zone.
            for idx, polygon in enumerate(self.polygons):
                zone_data = self.zone_interaction_dict[idx]['track_info']
                if polygon.contains(point):
                    # If the track ID is not already in the zone, record its entry time.
                    if track_id not in zone_data:
                        zone_data[track_id] = current_time
                else:
                    # Remove track IDs that are no longer in the zone.
                    if track_id in zone_data:
                        del zone_data[track_id]

        # For each zone, filter out track IDs that haven't been present long enough.
        for idx in range(len(self.polygons)):
            zone_data = self.zone_interaction_dict[idx]['track_info']
            valid_track_ids = [
                tid for tid, entry_time in zone_data.items() 
                if (current_time - entry_time) >= self.min_presence_time
            ]
            count = len(valid_track_ids)
            # Update the window deques for smoothing.
            self.window_data[idx]["count"].append(count)
            self.window_data[idx]["track_ids_list"].append(valid_track_ids)
            try:
                most_frequent_count = mode(self.window_data[idx]["count"])
            except Exception:
                most_frequent_count = count
            self.zone_interaction_dict[idx]['most_frequent_count'] = most_frequent_count
            if most_frequent_count > 0:
                # Combine all track ID lists from the window.
                all_track_ids = [tid for sublist in self.window_data[idx]["track_ids_list"] for tid in sublist]
                counter = Counter(all_track_ids)
                # Get the top most_frequent_count track IDs.
                most_frequent_track_ids = [tid for tid, cnt in counter.most_common(most_frequent_count)]
                self.zone_interaction_dict[idx]['most_frequent_track_ids'] = most_frequent_track_ids
            else:
                self.zone_interaction_dict[idx]['most_frequent_track_ids'] = []
        return self._convert_to_dict(self.zone_interaction_dict)

    def _get_center(self, bbox):
        """Calculate the center of the bounding box [x1, y1, x2, y2]."""
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = bbox[3]  # using the bottom edge; adjust if needed
        return (x_center, y_center)

    def _convert_to_dict(self, d):
        """Recursively convert defaultdicts to standard dicts."""
        if isinstance(d, defaultdict):
            d = {k: self._convert_to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: self._convert_to_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            d = [self._convert_to_dict(i) for i in d]
        return d
