import os
import sys
import logging
import datetime
import argparse
import cv2
from pymongo import MongoClient
from bson.objectid import ObjectId

# MongoDB connection config
MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://sentinel:sentinelMongo_test123@localhost:27017/sentinel_fm")
MONGO_DB = os.environ.get("MONGODB_DBNAME", "sentinel_fm")
METADATA_COLLECTION = "metadata"
CAMERA_CONFIG_COLLECTION = "camera_config"
SERVICES_COLLECTION = "services"
CAMERAS_COLLECTION = "cameras"

################################################################################
# Cropping Service
################################################################################
class EntryExitCropStorageService:
    """
    Processes metadata from the database to create and store cropped images
    for each detected object.
    """
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        self.crop_dir = os.path.join(project_root, "results", "track_ids")
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[MONGO_DB]
        print("EntryExitCropStorageService initialized and connected to the database.")

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
                zone_name = "no-zone"
                track_id_dir = os.path.join(
                    self.crop_dir, date_str, str(device_id), zone_name, str(track_id)
                )
                os.makedirs(track_id_dir, exist_ok=True)
                file_name = f"{file_time_str}-{label}.jpg"
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

################################################################################
# Camera and Service Filter Logic
################################################################################
def get_entry_exit_cameras(mongo_client):
    """
    Returns a list of device_ids for cameras where
    any linked service has service_name == "Entry_Exit_Service".
    """
    db = mongo_client[MONGO_DB]
    eligible_ids = []
    # Step 1: Find all cameras
    for camera in db[CAMERAS_COLLECTION].find({}, {"_id": 1}):
        cam_oid = camera["_id"]

        # Step 2: Find matching camera_config with camera_oid == cam_oid
        config_doc = db[CAMERA_CONFIG_COLLECTION].find_one({"camera_oid": cam_oid})
        if not config_doc or "services" not in config_doc:
            continue

        found_entry_exit_service = False
        # Step 3: For each service in camera_config, check service_name in services collection
        for svc_obj in config_doc["services"]:
            service_oid = svc_obj.get("service")
            if not service_oid:
                continue
            service_doc = db[SERVICES_COLLECTION].find_one({"_id": service_oid})
            if service_doc and service_doc.get("service_name") == "Entry_Exit_Service":
                found_entry_exit_service = True
                break
        if found_entry_exit_service:
            eligible_ids.append(cam_oid)
    return eligible_ids

################################################################################
# Script Entry Point
################################################################################
if __name__ == "__main__":
    """
    Run the cropping service for every camera that:
    - Exists in cameras
    - Has a related service in camera_config.services having service_name == 'Entry_Exit_Service'
    """
    # Optional: make time window CLI argument
    parser = argparse.ArgumentParser(description="Run the crop storage service for Entry_Exit_Service cameras.")
    parser.add_argument(
        '--minutes_ago',
        type=int,
        required=False,
        default=10,
        help='Time window in minutes for processing metadata (default: 10)'
    )
    args = parser.parse_args()

    # logger = setup_logging("filtered_crop_service")
    print("filtered_crop_service")
    mongo_client = MongoClient(MONGO_URI)
    service = EntryExitCropStorageService()

    # Find eligible device_ids
    entry_exit_device_ids = get_entry_exit_cameras(mongo_client)
    # logger.info(f"Eligible cameras with Entry_Exit_Service: {entry_exit_device_ids}")
    print(f"Eligible cameras with Entry_Exit_Service: {entry_exit_device_ids}")

    for device_id in entry_exit_device_ids:
        service.run(minutes_ago=args.minutes_ago, device_id=device_id)