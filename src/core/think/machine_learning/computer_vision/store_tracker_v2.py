
# -*- coding: utf-8 -*-
'''
Created on 12-June-2024 21:51
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''


from collections import deque, Counter, defaultdict
import copy, time
from src.database.schemas.store_zone_schema import StoreZoneState

class StoreTracker:
    def __init__(self, device_names, window_size=30):
        self.window_size = window_size
        self.window_data =  defaultdict(lambda: deque(maxlen=self.window_size))

        self.device_names = device_names

        self.store_counts = defaultdict()
        self.store_counts["entrance"] = {
                                    "customer": list(),
                                    "employee": list()
                                        }
        self.store_counts["help_desk"] = {
                                    "customer": list(),
                                    "employee": list(),
                                    "time_delta": 30 # sec
                                }
        self.store_counts["accessories"] = {
                                    "customer": list(),
                                    "employee": list(),
                                    "time_delta": 8 # sec

                                }
        self.store_counts["mobile"] = {
                                    "customer": list(),
                                    "employee": list(),
                                    "time_delta": 10 # sec
                                }
        
        self.store_counts["playstation"] = {
                                    "customer": list(),
                                    "employee": list(),
                                    "time_delta": 3 # sec
                                }
        self.store_counts["laptop"] = {
                                    "customer": list(),
                                    "employee": list(),
                                    "time_delta": 5 # sec
                                }
        

        self.last_store_counts = copy.deepcopy(self.store_counts)
        self.default_store_counts = copy.deepcopy(self.store_counts)


        self.zonal_state = {}
        
        self.last_entered_ids = deque(maxlen=self.window_size)
        self.unzoned_ids =  deque(maxlen=self.window_size)


    def update(self, data):
        
        # print("\n*************************")
        current_time = time.time()
        for device_name in self.device_names:
            if device_name in data and "data" in data[device_name] and "kpi" in data[device_name]["data"]:

                # print("\n")
                # print(device_name, data[device_name]['data']['kpi'])
# 
                if 'zone_name_list' in  data[device_name]['data']['kpi'] and data[device_name]["data"]['kpi']["zone_name_list"][0] == "entrance":

                    if len(data[device_name]["data"]['kpi']['entry_exit_kpi']['customer']['enter']['track_ids']) > 0:
                        
                        entered_track_ids = data[device_name]["data"]['kpi']['entry_exit_kpi']['customer']['enter']['track_ids']

                        for track_id in entered_track_ids:
                            if track_id not in self.last_entered_ids:

                                print("entry##############")
                                self.last_entered_ids.append(track_id)
                                self.unzoned_ids.append({track_id: current_time})
                                self.store_counts["entrance"]["customer"].append(track_id)

                if 'zone_name_list' in  data[device_name]['data']['kpi'] and data[device_name]["data"]['kpi']["zone_name_list"][0] != "entrance":
                    

                    zone_name = data[device_name]["data"]['kpi']["zone_name_list"][0]

                    zone_name = zone_name.replace(" ", "_")

                    most_frequent_count = data[device_name]["data"]['kpi']["zone_interaction_dict"]["customer"][0]["most_frequent_count"]
                    track_ids = data[device_name]["data"]['kpi']["zone_interaction_dict"]["customer"][0]["updated_track_ids"]

                    if len(self.unzoned_ids) > 0:
                            
                        if most_frequent_count > self.zonal_state[zone_name]:
                            # found new track id in zone
                            for track_id in track_ids:
                                if len(self.unzoned_ids) > 0:


                                    if zone_name in self.store_counts and track_id in self.store_counts[zone_name]['customer']:
                                        continue

                                    unzoned_id = self.unzoned_ids[0]
                                    track_id_time = list(unzoned_id.values())[0]
                                    _track_id = list(unzoned_id.keys())[0]
                                    
                                    time_delta = time.time() - track_id_time

                                    if self.store_counts[zone_name]["time_delta"] < time_delta:
                                        unzoned_id = self.unzoned_ids.popleft()

                                        self.store_counts[zone_name]['customer'].append(_track_id)

                    # for track_id in self.store_counts[zone_name]['customer']:
                    #     if track_id not in track_ids:
                    #         self.store_counts[zone_name]['customer'].remove(track_id)

                    self.zonal_state[zone_name] = most_frequent_count
                            

                #     self.window_data[device_name].append(data[device_name]["data"]['kpi']["zone_name_list"]["entry_exit_kpi"])

        # print("\n")

        # print(self.last_store_counts)
        if self.store_counts != self.last_store_counts and self.store_counts != self.default_store_counts:
            self.last_store_counts = copy.deepcopy(self.store_counts)
            self.store_counts = copy.deepcopy(self.default_store_counts)

            # store self.store_counts in database 
            StoreZoneState.store(self.last_store_counts)

            # print("copied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # print(self.store_counts)


        