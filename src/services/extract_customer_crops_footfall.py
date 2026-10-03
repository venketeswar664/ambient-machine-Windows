import os
import cv2
import datetime
import argparse
from pymongo import MongoClient
from bson.objectid import ObjectId

def extract_crops():
    # Setup MongoDB
    MONGO_URI = "mongodb://sentinel:sentinelMongo_test123@10.8.0.26:27017/sentinel_warehouse?directConnection=true&authSource=admin"
    print(f"Connecting to MongoDB at {MONGO_URI}")
    client = MongoClient(MONGO_URI)
    db = client["sentinel_warehouse"]

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, "results", "footfall")
    
    # Target Device
    target_device = ObjectId("69ba968fd51e7a4ac615a026")

    # 8 April to 13 April inclusive
    start_date = datetime.datetime(2026, 6, 25, 0, 0, 0)
    end_date = datetime.datetime(2026, 6, 25, 23, 59, 59, 999999)

    print(f"Extracting customer crops for device {target_device} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Output directory: {output_dir}")

    # STEP 1: Fetch valid track_ids from `track_id_metadata`
    print("Querying track_id_metadata collection for valid tracks...")
    track_query = {
        "start_time": {"$gte": start_date, "$lt": end_date},
        "device": target_device,
        "label": "customer"
    }

    valid_tracks = {}
    track_cursor = db.track_id_metadata.find(track_query)
    for doc in track_cursor:
        track_id = doc.get("track_id")
        start_time_obj = doc.get("start_time")
        if track_id and start_time_obj:
            # We strictly need the first timestamp for creating the folder name
            valid_tracks[track_id] = start_time_obj.strftime("%Y-%m-%d_%H-%M-%S")
            
    num_valid_tracks = len(valid_tracks)
    print(f"Found {num_valid_tracks} customer track IDs from track_id_metadata.")

    if num_valid_tracks == 0:
        print("No valid tracks found. Exiting.")
        return

    # STEP 2: Process `metadata` frames strictly for the valid tracks
    print("Querying metadata collection for all frames...")
    # NOTE: Sometimes the metadata 'time_stamp' can be slightly outside the track's start_time/end_time bounds,
    # but we can query by the general date range or slightly larger bounds to ensure we capture all crops.
    metadata_query = {
        "device": target_device,
        "time_stamp": {"$gte": start_date - datetime.timedelta(minutes=5),
                       "$lt": end_date + datetime.timedelta(minutes=5)},
        "track_ids_info": {"$exists": True, "$ne": {}}
    }
    
    total_docs = db.metadata.count_documents(metadata_query)
    print(f"Found {total_docs} metadata documents in the specified date range.")
    
    cursor = db.metadata.find(metadata_query).sort("time_stamp", 1)

    processed_count = 0
    saved_crops = 0

    for doc in cursor:
        processed_count += 1
        if processed_count % 5000 == 0:
            print(f"Processed {processed_count} / {total_docs} documents. Saved crops so far: {saved_crops}")

        raw_frame_path = doc.get("raw_frame_path")
        if not raw_frame_path:
            continue

        if not os.path.isabs(raw_frame_path):
            raw_frame_path = os.path.join(project_root, raw_frame_path.lstrip("./"))

        if not os.path.exists(raw_frame_path):
            continue

        track_ids_dict = doc.get("track_ids_info", {})
        if not track_ids_dict:
            continue

        timestamp_obj = doc.get("time_stamp")
        if not timestamp_obj:
            continue
            
        date_str = timestamp_obj.strftime("%Y-%m-%d")
        frame = None

        for track_id, info in track_ids_dict.items():
            # Check if this track is within our strictly filtered valid track list!
            if track_id not in valid_tracks:
                continue
                
            bbox = info.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            first_seen_str = valid_tracks[track_id]
            
            # Target output folder mimicking the image structure:
            # footfall/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS_trackId/
            track_folder_name = f"{first_seen_str}_{track_id}"
            track_dir = os.path.join(output_dir, date_str, track_folder_name)
            os.makedirs(track_dir, exist_ok=True)
            
            # File name mimicking the image structure:
            # YYYY-MM-DD_HH-MM-SS-MS-customer.jpg
            timestamp_ms_str = timestamp_obj.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            file_name = f"{timestamp_ms_str}-customer.jpg"
            output_path = os.path.join(track_dir, file_name)
            
            if os.path.exists(output_path):
                continue
                
            # Lazy load the frame
            if frame is None:
                frame = cv2.imread(raw_frame_path)
                if frame is None:
                    break   # Move to next document if the frame can't be read

            x1, y1, x2, y2 = map(int, bbox)
            y1, y2 = max(0, y1), max(0, y2)
            x1, x2 = max(0, x1), max(0, x2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            if cv2.imwrite(output_path, crop):
                saved_crops += 1

    print(f"\nExtraction complete! Processed {processed_count} metadata documents.")
    print(f"Total customer crops saved: {saved_crops}")

if __name__ == "__main__":
    extract_crops()