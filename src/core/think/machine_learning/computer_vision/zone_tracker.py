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


class ZoneTracker(Module):
    def __init__(self, _config):
        Module.__init__(self, _config)
        self.config = _config
        self.window_size = self.config["window"]

        self.image_width = None
        self.image_height = None
        self.camera_name = None

        self.camera = self.config["camera"]

        self.zonal_data = deque(dict(), maxlen=self.window_size)

        # set zone and polygons
        zones = self.config["zones"]
        self.zone_dict = {}
        self.set_zone(zones)

        self.exit_dict = {}

        # print("zone_dict: ", self.zone_dict)

    def set_image_size(self):
        frame = self.camera.rgb_img
        self.camera_name = self.camera.device_name
        self.image_width, self.image_height = frame.shape[:2]

    def set_zone(self, zones):

        for key in zones:
            zone = zones[key]
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

            zone_name = zone.name
            zone_type = zone.zone_type
            
            self.zone_dict[key] = {"shape": shape,
                                         "type": zone_type,
                                         "name": key}
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

    def process(self, input_dict) -> dict:
        
        boxes = input_dict["boxes"]
        track_ids = input_dict["track_ids"]
        current_time = input_dict["time_stamp"]
        frame = input_dict["frame"]


        output_dict = {}
        movement_dict = {}
        plotted_frame = frame
        zonal_dict = {}

        if track_ids:
            if self.image_height or self.image_width is None:
                self.set_image_size()
            
            # get track_ids in the zones
            zonal_dict = self.track_ids_in_roi(boxes, track_ids, current_time)

            # track entry exit of the zone
            movement_dict = self.track_movement()

            plotted_frame = plot.plot_dict(frame, zonal_dict)
            plotted_frame = plot.plot_dict(plotted_frame, movement_dict, starting_point='top_left')


        output_dict["frame"] = plotted_frame
        output_dict["zone"] = zonal_dict
        output_dict["movement"] = movement_dict

        # print("zonal_dict :", zonal_dict)

        return output_dict
    
    def track_movement(self):
        # print(f"{self.camera_name} zonal_data", self.zonal_data)


        last_seen = {}  # Dictionary to track last known location of track_ids
        transitions = {
            "entry": {},
            "exit": {}
        }  # List to store transitions from lab/entry to zone
        for record in self.zonal_data:
            time = record['time']
            current_track_ids = []


            for area_name in record:
                if area_name not in ["zone", "time"]:
                    area_track_ids = record.get(area_name, [])

                    # Check if there is a lab entry
                    if area_track_ids:
                        # Mark track_ids seen

                        # current track ids 
                        current_track_ids = current_track_ids + area_track_ids

                        for track_id in area_track_ids:

                            # if area_name not in self.exit_dict:
                            #     self.exit_dict[area_name] = []

                            # if track_id not in self.exit_dict[area_name]:

                            if last_seen.get(track_id) == 'zone':

                                # print("area_name", area_name, track_id)

                                if area_name in transitions["exit"]:

                                    if track_id not in transitions["exit"][area_name]:

                                        transitions["exit"][area_name].append(track_id)

                                else:
                                
                                    transitions["exit"][area_name] = [track_id]      

                                # self.exit_dict[area_name].append(track_id)
                                
                
                            last_seen[track_id] = area_name

                
                if area_name == "zone":
                    zone_track_id = record.get(area_name, [])


                    # Check if there is a zone entry
                    if zone_track_id:

                        # current track ids 
                        current_track_ids = current_track_ids + zone_track_id

                        # Check if any track_id is transitioning from lab to zone
                        for track_id in zone_track_id:
                            
                            last_area_name = last_seen.get(track_id)
                            if last_area_name  != "zone" and last_area_name != None:

                                if last_area_name in transitions["entry"]:
                                    if track_id not in transitions["entry"][last_area_name]:
                                        transitions["entry"][last_area_name].append(track_id)
                                else:
                                    transitions["entry"][last_area_name] = [track_id]


                            # Update last seen to zone if no prior lab entry
                            last_seen[track_id] = 'zone'


                # remove_track_ids = list(set(last_seen.keys()) - set(current_track_ids))

                # print("last_seen ", list(set(last_seen.keys())))
                # print("current_track_ids ", set(current_track_ids))
                # print("removing ", remove_track_ids)

                # for track_id in remove_track_ids:
                #     del last_seen[track_id]

        return transitions

    
    def track_ids_in_roi(self, boxes, track_ids, current_time):
        
        zonal_dict = {"time": current_time}
        for i, track_id in enumerate(track_ids):
            bbox = boxes[i]
            current_position = self._get_center(bbox)
            point = Point(current_position)


            for zone_name in self.zone_dict:
                polygon = self.zone_dict[zone_name]["shape"]

                if polygon.contains(point):

                    if zone_name in zonal_dict:
                        zonal_dict[zone_name].append(track_id)
                    else:
                        zonal_dict[zone_name] = [track_id]

        self.zonal_data.append(zonal_dict)

        return zonal_dict

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

