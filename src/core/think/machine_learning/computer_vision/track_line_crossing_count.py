# -*- coding: utf-8 -*-
'''
Created on 31-May-2024 18:36
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''


import cv2
import numpy as np
from collections import deque
from shapely.geometry import LineString, Point
from src.utils.plot import _get_center

class TrackCrossingCounter:
    def __init__(self, line_start, line_end, window=5):
        self.line = LineString([line_start, line_end])
        self.cross_counts = {
            'customer': {
                'enter': {
                    'count': 0,
                    'track_ids': []
                },
                'exit': {
                    'count': 0,
                    'track_ids': []
                }
            },
            'employee': {
                'enter': {
                    'count': 0,
                    'track_ids': []
                },
                'exit': {
                    'count': 0,
                    'track_ids': []
                }
            }
        }
        self.track_positions = {}
        self.frame_window = window

    def track_crossing(self, boxes, track_ids, labels):
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            label = labels[i]  # 0 for customer, 1 for employee
            label_key = 'customer' if label == 0 else 'employee'
            current_position = _get_center(bbox)
            if track_id not in self.track_positions:
                self.track_positions[track_id] = deque(maxlen=self.frame_window)
            self.track_positions[track_id].append(current_position)

            if len(self.track_positions[track_id]) == self.frame_window:
                # Check crossing
                positions = list(self.track_positions[track_id])
                last_line = LineString([positions[0], positions[-1]])
                if last_line.crosses(self.line):
                    # Determine direction
                    if self._is_bottom_right_to_top_left(positions[0], positions[-1]):
                        if track_id not in self.cross_counts[label_key]['enter']['track_ids']:
                            self.cross_counts[label_key]['enter']['count'] += 1
                            self.cross_counts[label_key]['enter']['track_ids'].append(track_id)
                    else:
                        if track_id not in self.cross_counts[label_key]['exit']['track_ids']:
                            self.cross_counts[label_key]['exit']['count'] += 1
                            self.cross_counts[label_key]['exit']['track_ids'].append(track_id)

        # Prepare the result and reset for the next iteration
        result = self.cross_counts.copy()
        self._reset()
        return result

    def _reset(self):
        """
        Reset the tracking data for the next iteration.
        """
        # self.track_positions.clear()
        self.cross_counts = {
            'customer': {
                'enter': {
                    'count': 0,
                    'track_ids': []
                },
                'exit': {
                    'count': 0,
                    'track_ids': []
                }
            },
            'employee': {
                'enter': {
                    'count': 0,
                    'track_ids': []
                },
                'exit': {
                    'count': 0,
                    'track_ids': []
                }
            }
        }

    def _get_center(self, bbox):
        """
        Calculate the center of the bounding box.
        bbox is in the format [x1, y1, x2, y2].
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = bbox[3]
        return (x_center, y_center)

    def _is_bottom_right_to_top_left(self, start, end):
        """
        Determine if the direction of movement is from bottom-right to top-left (enter).
        """
        return start[0] > end[0] and start[1] > end[1]

# Example usage
if __name__ == "__main__":
    line_start = (0, 480)  # Example line start point
    line_end = (640, 0)  # Example line end point

    counter = TrackCrossingCounter(line_start, line_end)

    boxes = [(100, 100, 150, 150), (200, 200, 250, 250), (300, 300, 350, 350), (400, 400, 450, 450)]
    track_ids = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]  # 0 for customer, 1 for employee

    # Simulate updating the counter with data
    cross_counts = counter.track_crossing(boxes, track_ids, labels)
    print(cross_counts)
