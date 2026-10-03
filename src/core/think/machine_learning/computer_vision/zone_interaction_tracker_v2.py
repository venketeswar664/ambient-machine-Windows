# -*- coding: utf-8 -*-
'''
Created on 29-May-2024 09:28
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''

import cv2
import numpy as np
from collections import defaultdict, deque, Counter
from shapely.geometry import Point, Polygon, LineString
from src.utils import plot

import time
import pprint
import math
import copy
from statistics import mode
from src.core.modules.module import Module


class ZoneInteractionTracker(Module):
    def __init__(self, _config) -> None:
        Module.__init__(self, _config)
        self.config = _config
        self.window_size = self.config["window"]
        camera = self.config["camera"]


        self.image_width = None
        self.image_height = None


        self.last_tracked_ids = set()

        self.window_data = defaultdict(lambda:{"count": deque(set(), maxlen=self.window_size), \
                                                "track_ids_list": deque(list(), maxlen=self.window_size)})
        self.last_most_frequent_count = None
        self.last_most_frequent_track_ids = None

        self.track_id_record = {}

        # set zone and polygons
        zones = self.config["zones"]
        self.polygons = []
        self.zone_name_list = []
        zone_name, zone_type = self.set_zone(zones)

        self.set_image_size(camera)
        self.zone_interaction_dict = defaultdict(lambda:{
            'track_ids': list(),
            'zone_frequent_track_ids': list(),
            'zone_updated_track_ids': list(),
            "zone_name": zone_name,
            "zone_type": zone_type

        })


    def set_zone(self, zones):

        for zone in zones:
            vertices = zone.roi
            vertices_points = [int(x) for x in vertices[0]]
            vertices_points = [vertices_points[i:i+2] for i in range(0, len(vertices_points), 2)]
            # print("vertices_points", vertices_points)

            if len(vertices_points) > 2:
                # Store as polygon
                shape = Polygon(vertices_points)
            elif len(vertices_points) == 2:
                # Store as line
                shape = LineString(vertices_points)
            else:
                # Not enough points to form a line or polygon
                shape = None
            

            self.polygons.append(shape)
            self.zone_name_list.append(zone.name)
            zone_name = zone.name
            zone_type = zone.zone_type
            return zone_name, zone_type
            

    def set_image_size(self, camera):
        frame = camera.rgb_img
        self.image_width, self.image_height = frame.shape[:2]

    def process(self, input_dict) -> dict:
        
        boxes = input_dict["boxes"]
        track_ids = input_dict["track_ids"]

        output_dict = self.update(boxes, track_ids)
        return output_dict
    
    def plot(self, input_dict) -> None:

        frame = input_dict["frame"]
        boxes = input_dict["boxes"]
        track_ids = input_dict["track_ids"]
        labels = input_dict["labels"]
        updated_track_ids = input_dict["zone_updated_track_ids"]

        # plot track id and its positions
        plotted_frame = plot.plot_point_and_trackid(frame, boxes, track_ids, labels)
        
        # plot shapes
        plotted_frame = plot.plot_shapes(plotted_frame, self.polygons)

        # plot kpis
        plot_dict = {}
        plot_dict["track_ids"] = track_ids
        plot_dict["updated_track_ids"] = updated_track_ids

        plotted_frame = plot.plot_dict(plotted_frame, plot_dict)


    def update(self, boxes, track_ids):


        current_time = time.time()
        current_tracked_ids = set(track_ids)

        # Handle track IDs that were in polygons in the last frame but are missing in the current frame
        missing_ids = self.last_tracked_ids - current_tracked_ids
        for track_id in missing_ids:
            self._handle_missing_track_id(track_id, current_time)

        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            current_position = self._get_center(bbox)
            point = Point(current_position)
            

            for polygon_index, polygon in enumerate(self.polygons):
                zone_name = self.zone_name_list[polygon_index]
                if polygon.contains(point):
                    if track_id not in self.zone_interaction_dict[zone_name]['track_ids']:
                        self.zone_interaction_dict[zone_name]['track_ids'].append(track_id)

                else:
                    if track_id in self.zone_interaction_dict[zone_name]['track_ids']:
                        self.zone_interaction_dict[zone_name]['track_ids'].remove(track_id)



        
        for polygon_index, polygon in enumerate(self.polygons):
            zone_name = self.zone_name_list[polygon_index]

            track_id_list = self.zone_interaction_dict[zone_name]['track_ids']
            track_id_count = len(track_id_list)

            self.window_data[polygon_index]["count"].append(track_id_count)
            self.window_data[polygon_index]["track_ids_list"].append(track_id_list)

            most_frequent_count =  mode(self.window_data[polygon_index]["count"])
            # self.zone_interaction_dict[zone_name]['most_frequent_count'] = most_frequent_count


            if most_frequent_count > 0:

                all_track_ids = np.array(self.window_data[polygon_index]["track_ids_list"]).ravel()

                counter = Counter(all_track_ids)
                zone_frequent_track_ids = list(counter.keys())[:most_frequent_count]
                self.zone_interaction_dict[zone_name]['zone_frequent_track_ids'] = copy.deepcopy(zone_frequent_track_ids)
                


                # track_id fixing 
                updated_track_id = copy.deepcopy(zone_frequent_track_ids)
                if self.last_most_frequent_count != None:
                    # check condition for track_id split
                    if len(self.track_id_record) > 0:   
                        tracked_ids = set(self.track_id_record.keys()).intersection(set(zone_frequent_track_ids))
                    
                        # check if recorded ids intersection with current track_ids
                        if len(tracked_ids):

                            for track_id_1 in tracked_ids:
                                updated_track_id[zone_frequent_track_ids.index(track_id_1)] = self.track_id_record[track_id_1]
                            


                    if self.last_most_frequent_count == most_frequent_count:
                        if self.last_most_frequent_track_ids != zone_frequent_track_ids:

                            command_ids = list(set(self.last_most_frequent_track_ids).intersection(set(zone_frequent_track_ids)))
                            old_ids = list(set(self.last_most_frequent_track_ids) - set(command_ids))
                            new_ids = list(set(zone_frequent_track_ids) - set(command_ids))

                            for old_id, new_id in zip(old_ids, new_ids):

                                if new_id not in list(self.track_id_record.values()):
                                    self.track_id_record[new_id] = old_id

                                    updated_track_id[zone_frequent_track_ids.index(new_id)] = self.track_id_record[new_id]


                self.last_most_frequent_count = copy.deepcopy(most_frequent_count)
                self.last_most_frequent_track_ids = copy.deepcopy(zone_frequent_track_ids)

                # print("updated zone_updated_track_ids", zone_frequent_track_ids)
                # print("self.track_id_record", self.track_id_record)

                self.zone_interaction_dict[zone_name]['zone_updated_track_ids'] = copy.deepcopy(zone_frequent_track_ids)
                # self.zone_interaction_dict[label_key][polygon_index]['track_ids'] = zone_frequent_track_ids

            else:
                self.zone_interaction_dict[zone_name]['zone_frequent_track_ids'] = []

                self.zone_interaction_dict[zone_name]['zone_updated_track_ids'] = []
                self.track_id_record = {}

                    
        # Update last tracked IDs
        self.last_tracked_ids = current_tracked_ids    

        # Convert defaultdict to regular dict for pretty printing + =
        zone_interaction_dict = self._convert_to_dict(self.zone_interaction_dict[zone_name])

        print("zone_interaction_dict", zone_interaction_dict)
        return zone_interaction_dict

    def _handle_missing_track_id(self, track_id, current_time):
        """
        Handle track IDs that are missing in the current frame.
        """
     
        for polygon_index in range(len(self.polygons)):
            zone_name = self.zone_name_list[polygon_index]

            if track_id in self.zone_interaction_dict[zone_name]['track_ids']:
                self.zone_interaction_dict[zone_name]['track_ids'].remove(track_id)


    def _get_center(self, bbox):
        """
        Calculate the center of the bounding box.
        bbox is in the format [x1, y1, x2, y2].
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = bbox[3]
        return (x_center, y_center)

    def _calculate_distance(self, pos1, pos2):
        """
        Calculate the Euclidean distance between two points.
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _convert_to_dict(self, d):
        """
        Convert a nested defaultdict to a regular dict and format durations.
        """
        if isinstance(d, defaultdict):
            d = {k: self._convert_to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: self._convert_to_dict(v) for k, v in d.items()}
        return d
