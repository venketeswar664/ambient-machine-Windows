# -*- coding: utf-8 -*-
'''
Created on 27-June-2024 02:26
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''

from collections import defaultdict
import threading
from datetime import datetime
from statistics import mode
from src.core.modules.module import Module
from src.database.schemas.kpi_schema_v2 import Trackid_Metadata

class LabelTracker(Module):
    def __init__(self, _config) -> None:
        """
        Initialize the LabelTracker class to manage track IDs and their labels over time.
        
        Args:
            time_limit (int, optional): Time limit in seconds for retaining track ID data.
                                       Defaults to 10 seconds.
        """
        Module.__init__(self, _config)
        self.config = _config

        self.time_limit = self.config["time_limit"]
        self.window_data = defaultdict(lambda: {"label_list": [], "label": None, "start_time": None, "end_time": None, "db_ref": None})


    def process(self, input_dict) -> dict:

        labels = input_dict["labels"]
        track_ids = input_dict["track_ids"]

        self.update_labels(track_ids, labels)

        output_dict = {}
        output_dict["labels"] = self.get_label(track_ids)

        return output_dict



    def update_labels(self, track_ids, labels):
        """
        Update labels for given track IDs and manage label history and update times.

        Args:
            track_ids (list): List of track IDs predicted in the frame.
            labels (list): Corresponding list of labels (0 or 1) for each track ID.
        """
        current_time = datetime.now()
        
        for track_id, label in zip(track_ids, labels):
            if track_id in self.window_data:
                # Append the new label to label_list
                self.window_data[track_id]["label_list"].append(label)
                # Update the last update time for the track ID
                self.window_data[track_id]["end_time"] = current_time
                # Update the label to the most frequent value in label_list
                self.window_data[track_id]["label"] = mode(self.window_data[track_id]["label_list"])
            else:
                # Initialize label_list with the first label
                self.window_data[track_id]["label_list"] = [label]
                self.window_data[track_id]["label"] = label
                self.window_data[track_id]["start_time"] = current_time
                self.window_data[track_id]["end_time"] = current_time

        # update track_ids_on db
        self.update_track_ids_in_db(track_ids)
        
        # Check and remove track IDs that have exceeded the time limit
        self.check_time_limit(track_ids)

    def update_track_ids_in_db(self, track_ids):

        updater = threading.Thread(target=self.push_data, args=(track_ids, ))
        updater.start()

    def push_data(self, track_ids):
        """iterate through all the track_ids and create a document if the track_id does not exist in self.window_data and store track_id
            and labels in track_id_metadata collection and store the document in to the db and documents variable in self.window_data.
            If the track_id is in the self.window_data then it takes the document from self.window_data and update the label in the 
            database 

        Args:
            track_ids (list): list of track_id
        """

        for track_id in track_ids:
            track_data = self.window_data[track_id]
            label = track_data["label"]
            current_time = track_data["end_time"]

            if track_data["db_ref"]:
                # Update the label and end_time in the database
                track_data["db_ref"].update(
                    set__track_id_dict={"label": label},
                    set__end_time=current_time
                )
                print("update metadata")

            else:
                # Create a new document in the database
                new_document = Trackid_Metadata(
                    track_id=track_id,
                    track_id_dict={"label": label},
                    start_time=current_time,
                    end_time=current_time,
                    evidence_path=track_data.get("evidence_path", "")
                )
                new_document.save()

                print("saved track_id metadata", track_ids)
                # Store the document reference in window_data
                self.window_data[track_id]["db_ref"] = new_document

    def get_label(self, track_ids):
        """
        Retrieve the most recent label for each track ID.

        Args:
            track_ids (list): List of track IDs for which labels are required.
        
        Returns:
            labels (list): List of labels corresponding to the input track IDs.
        """
        labels = []
        for track_id in track_ids:
            if track_id in self.window_data:
                # Retrieve the current label for the track ID
                labels.append(self.window_data[track_id]["label"])
            else:
                labels.append(None)  # Handle case where track ID is not found

        return labels

    def check_time_limit(self, current_track_ids):
        """
        Check and remove track IDs that have exceeded the time limit for retention.
        
        Args:
            current_track_ids (list): List of currently tracked track IDs.
        """
        current_time = datetime.now()
        outdated_track_ids = []

        for track_id in list(self.window_data.keys()):
            if track_id not in current_track_ids:
                last_update_time = self.window_data[track_id]["end_time"]
                # Remove track ID if it has not been updated within the time limit
                if (current_time - last_update_time).total_seconds() > self.time_limit:
                    outdated_track_ids.append(track_id)

        for track_id in outdated_track_ids:
            # Store labels in database before deleting

            del self.window_data[track_id]
