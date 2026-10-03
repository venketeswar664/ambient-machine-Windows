# src/services/store_crop_service.py

import os
import sys
import argparse
import datetime
import cv2
from crontab import CronTab
from pymongo import MongoClient
from dateutil.parser import isoparse
from bson.objectid import ObjectId

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://sentinel:sentinelMongo_test123@10.8.0.26:27018/sentinel_warehouse?directConnection=true")
MONGO_DB = os.environ.get("MONGODB_DBNAME", "sentinel_warehouse")
METADATA_COLLECTION = "metadata"

#################################################################################
# PART 1: THE SERVICE THAT PERFORMS THE CROPPING
#################################################################################
class CropStorageService:
    """
    (Production Version) Processes metadata from the database to create and store
    cropped images for each detected object.
    """
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        self.crop_dir = os.path.join(project_root, "results", "track_ids")
        
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[MONGO_DB]
        print("CropStorageService initialized and connected to the database.")

    def store_crops_from_metadata(self, metadata_doc):
        raw_frame_path = metadata_doc.get("raw_frame_path")
        track_ids_dict = metadata_doc.get("track_ids_info", {})
        device_id = metadata_doc.get("device")
        timestamp_obj = metadata_doc.get("time_stamp") 

        if not all([raw_frame_path, track_ids_dict, device_id, timestamp_obj]):
            return
        
        if not os.path.isabs(raw_frame_path):
             project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             raw_frame_path = os.path.join(project_root, raw_frame_path.lstrip("./"))
        
        if not os.path.exists(raw_frame_path):
            print(f"[ERROR] Raw frame not found at path: {raw_frame_path}.")
            return
        
        frame = cv2.imread(raw_frame_path)
        if frame is None:
            return
        
        date_str = timestamp_obj.strftime("%Y-%m-%d")

        for track_id, obj_info in track_ids_dict.items():
            label_id = obj_info.get("label")
            label_name = obj_info.get("label_name", "unknown")
            
            if label_id != 0 and str(label_name).lower() != "customer":
                continue

            bbox = obj_info.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            try:
                label = obj_info.get("label_name", "unknown")
                file_time_str = timestamp_obj.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                
                instance_dict = obj_info.get("instance_dict", {})
                active_zones = []
                for z_name, z_info in instance_dict.items():
                    if z_info.get("location") == "inside":
                        active_zones.append(z_name)
                
                if not active_zones:
                    active_zones = ["no-zone"]
                
                file_name = f"{file_time_str}-{label}.jpg"
                
                for zone_name in active_zones:
                    track_id_dir = os.path.join(self.crop_dir, date_str, str(device_id), zone_name, str(track_id))
                    os.makedirs(track_id_dir, exist_ok=True)
                    
                    output_path = os.path.join(track_id_dir, file_name)
                    
                    success = cv2.imwrite(output_path, crop)
                    
                    if not success:
                        print(f"[ERROR] Failed to save crop for track_id {track_id} to {output_path}")
            except Exception as e:
                print(f"[ERROR] An unexpected error occurred while saving crop for track_id {track_id}: {e}")
    
    def run(self, minutes_ago=10, device_id=None):
        """Fetches and processes metadata for a specific device."""
        if not device_id:
            print("[ERROR] A device_id must be provided.")
            return

        device_name = str(device_id)
        try:
            camera_doc = self.db["cameras"].find_one({"_id": device_id})
            if camera_doc and "device_name" in camera_doc:
                device_name = camera_doc["device_name"]
        except Exception as e:
            print(f"[ERROR] Could not look up device_name for ID {device_id}: {e}")
        print(f"Starting crop job for device '{device_name}' (last {minutes_ago} minutes).")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(minutes=minutes_ago)
        
        query = {
            "device": device_id,
            "time_stamp": {"$gte": start_time, "$lt": end_time},
            "track_ids_info": {"$exists": True, "$ne": {}}
        }
        
        
        try:
            docs_list = list(self.db[METADATA_COLLECTION].find(query))
            count = len(docs_list)

            if count > 0:
                print(f"Found {count} documents with detections to process for '{device_name}'.")
                for doc in docs_list:
                    self.store_crops_from_metadata(doc)
            
            print(f"Crop job for '{device_name}' finished.")
        except Exception as e:
            print(f"[ERROR] A critical error occurred during the crop service run for '{device_id}': {e}")

#################################################################################
# PART 2: THE SERVICE THAT SETS UP THE CRON JOBS
#################################################################################
class CronManager:
    """
    (UPDATED) Manages cron jobs and creates custom, date-structured log files.
    """
    def __init__(self, logger):
        self.logger = logger
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[MONGO_DB]
        self.cron = CronTab(user=True)
        self.logger.info("CronManager initialized.")

    def get_distinct_device_ids(self):
        """Fetches a list of all unique device IDs from MongoDB."""
        try:
            device_ids = self.db[METADATA_COLLECTION].distinct("device")
            self.logger.info(f"Found distinct device IDs: {device_ids}")
            return device_ids
        except Exception as e:
            self.logger.error(f"Failed to fetch distinct device IDs: {e}")
            return []

    def setup_crop_cron_jobs(self):
        """
        Creates/updates cron jobs with log files saved to a custom, structured path.
        """
        self.logger.info("Setting up crop processing cron jobs...")
        device_ids = self.get_distinct_device_ids()
        if not device_ids:
            self.logger.warning("No devices found. No cron jobs will be created.")
            return
        # 1. Define the base directory for all logs.
        base_log_dir = "/logs"
        
        # 2. Get today's date to create the date-specific subfolder.
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        date_log_dir = os.path.join(base_log_dir, today_str)
        
        # 3. Ensure the date-specific directory exists.
        os.makedirs(date_log_dir, exist_ok=True)
        self.logger.info(f"Ensuring log directory exists: {date_log_dir}")
        
        # --- MODIFICATION END ---

        python_executable = sys.executable
        script_path = os.path.abspath(__file__)

        for dev_id in device_ids:
            # --- MODIFICATION START ---
            
            # 4. Look up the device_name from the cameras collection using the dev_id.
            device_name = "unknown_device" # Default name
            try:
                # Assuming your cameras collection is named 'cameras'
                camera_doc = self.db["cameras"].find_one({"_id": dev_id})
                if camera_doc and "device_name" in camera_doc:
                    device_name = camera_doc["device_name"]
                else:
                    self.logger.warning(f"Could not find device_name for ID: {dev_id}")
            except Exception as e:
                self.logger.error(f"Error looking up device_name for ID {dev_id}: {e}")
            
            # 5. Construct the full path for the new log file.
            log_file_path = os.path.join(date_log_dir, f"{device_name}.log")
            
            
            job_comment = f"CROP_JOB_FOR_{dev_id}"
            

            command = f'{python_executable} {script_path} --device_id "{dev_id}" >> {log_file_path} 2>&1'
            
            self.cron.remove_all(comment=job_comment)
            job = self.cron.new(command=command, comment=job_comment)
            job.minute.every(10) 
            self.logger.info(f"Scheduled job for device ID '{dev_id}', logging to {log_file_path}")

        self.cron.write()
        self.logger.info("Successfully wrote all changes to system crontab.")


#################################################################################
# PART 3: THE SCRIPT ENTRY POINT FOR THE CRON JOB
#################################################################################
if __name__ == "__main__":
    """
    This block runs when the file is executed directly as a script.
    """
    parser = argparse.ArgumentParser(description="Run the crop storage service for a specific device.")
    parser.add_argument(
        '--device_id',
        type=str,
        required=True,
        help='The specific device reference_id (as a string) to process.'
    )
    args = parser.parse_args()

    service = CropStorageService()
    
    device_object_id = ObjectId(args.device_id)
    
    service.run(minutes_ago=10, device_id=device_object_id)