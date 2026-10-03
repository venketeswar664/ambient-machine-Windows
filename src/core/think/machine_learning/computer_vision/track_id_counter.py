# -*- coding: utf-8 -*-
'''
Created on 12-June-2024 21:51
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''


from collections import deque, Counter, defaultdict

class TrackIDTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.window_data = defaultdict(lambda: deque(maxlen=self.window_size))

    def count_track_ids(self, track_ids, labels):
        self.window_data["count"].append(len(track_ids))

        self.window_data["track_ids"].append(track_ids)
        self.window_data["labels"].append(labels)


        counter = Counter(self.window_data["count"])
        
        most_frequent_count = counter.most_common(1)[0][1] if counter else 0


        return most_frequent_count


