import os, glob, shutil
import json
import mongoengine as db
import pandas as pd
import datetime
import mongoengine as me
import cv2

# Silence all but fatal errors
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    # Older versions may not have this
    pass

from dotenv import load_dotenv
from urllib.parse import urlparse
import re
import sys
from pathlib import Path

# Add src path to sys.path
src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))
from src.database.schemas.sentinel_poc_schema import Metadata, TrackIDRecord, Cameras 
from urllib.parse import urlparse
from collections import defaultdict
from bson import ObjectId
from tqdm import tqdm

import sys, os, contextlib

@contextlib.contextmanager
def suppress_stderr():
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


load_dotenv()
print("Loaded environment variables from .env (if found).")

mongo_uri_from_env = os.getenv("MONGODB_URI")
db_name_from_env = os.getenv("DB_NAME")

# Determine effective URI and DB name
if mongo_uri_from_env:
    effective_mongo_uri = mongo_uri_from_env
    # Try to parse DB name from URI if DB_NAME env var is not set
    if not db_name_from_env:
        parsed_uri = urlparse(effective_mongo_uri)
        path_db_name = parsed_uri.path.lstrip('/')
        effective_db_name = path_db_name if path_db_name else DEFAULT_DB_NAME
    else:
        effective_db_name = db_name_from_env
else:
    effective_mongo_uri = DEFAULT_MONGO_URI
    effective_db_name = db_name_from_env if db_name_from_env else DEFAULT_DB_NAME
    print(f"Warning: MONGODB_URI not found in .env. Using default: {DEFAULT_MONGO_URI}")


print(f"Attempting to connect to MongoDB URI: {effective_mongo_uri}")
print(f"Database name: {effective_db_name}")

try:
    # Disconnect any existing default connection first to be safe in Jupyter
    me.disconnect_all() 
    me.connect(db=effective_db_name, host=effective_mongo_uri, UuidRepresentation='standard', alias='default')
    print("Successfully connected to MongoDB!")

except me.errors.MongoEngineConnectionError as e:
    print(f"MongoDB Connection Error: {e}")
    print("Please ensure MongoDB is running and accessible, and your .env file is correctly configured.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



# --- Define Time Window and Other Constants ---
# Using user-provided default times. Ensure they are parsed to aware UTC datetimes.
# default_start_time_str = "2025-05-28T14:30:00"
# default_end_time_str = "2025-05-28T15:30:00"
default_start_time_str = "2025-05-29T12:30:00"
default_end_time_str = "2025-05-29T13:30:00"

try:
    analysis_start_time_utc = datetime.datetime.fromisoformat(default_start_time_str).replace(tzinfo=datetime.timezone.utc)
    analysis_end_time_utc = datetime.datetime.fromisoformat(default_end_time_str).replace(tzinfo=datetime.timezone.utc)
    print(f"Analysis Time Window: {analysis_start_time_utc.isoformat()} to {analysis_end_time_utc.isoformat()}")
except ValueError as ve:
    print(f"Error parsing time strings: {ve}. Exiting.")
    # exit()

# --- Helper function for parsing zone timestamps ---
def parse_iso_timestamp_from_zone_key(ts_str):
    """Helper function to parse ISO timestamp strings from zone data keys."""
    if not ts_str: return None
    try:
        # Handle potential '+0000' from strftime %z on some systems if fromisoformat is strict
        if ts_str.endswith('+0000'):
            ts_str = ts_str[:-5] + '+00:00'
        elif ts_str.endswith('-0000'): # Should not happen for UTC but good to be safe
             ts_str = ts_str[:-5] + '-00:00'
        return datetime.datetime.fromisoformat(ts_str)
    except ValueError:
        # Fallback for older formats if necessary, but keys should be ISO an T separator
        try: # Example: 2023-10-26T10:30:00.123456
            return datetime.datetime.fromisoformat(ts_str.split('+')[0].split('-')[0]) # Attempt to parse without TZ
        except ValueError:
            print(f"Warning: Could not parse zone timestamp string with fromisoformat: '{ts_str}'")
            return None


# --- Part 1 & 2: Specific Camera Logic (Camera-018 and Camera-016) ---
print("\n--- Part 1 & 2: Specific Camera Filtering ---")

# This dictionary will store results for Camera-018 and Camera-016
# Structure: specific_camera_results[camera_name_str]["traveled_inside" OR "traveled_outside"][track_id] = [list of record_details_dict]
specific_camera_results = {
    "Camera-018": {"traveled_inside": defaultdict(list), "traveled_outside": defaultdict(list)},
    "Camera-016": {"traveled_inside": defaultdict(list), "traveled_outside": defaultdict(list)}
}

cameras_to_process_specifically = ["Camera-018", "Camera-016"]

for camera_device_name_filter in cameras_to_process_specifically:
    print(f"\nProcessing for {camera_device_name_filter}...")
    
    # Determine zone names based on camera
    if camera_device_name_filter == "Camera-018":
        inside_zone_label = "inside (exit)"
        outside_zone_label = "outside (entry)"
    elif camera_device_name_filter == "Camera-016":
        inside_zone_label = "inside (exit)"
        outside_zone_label = "outside (exit)"
    else:
        print(f"Warning: Zone labels not defined for {camera_device_name_filter}. Skipping.")
        continue

    # Fetch camera object
    cam_obj_for_filter = Cameras.objects(device_name=camera_device_name_filter).first()
    if not cam_obj_for_filter:
        print(f"Camera {camera_device_name_filter} not found in DB. Skipping.")
        continue

    # Query TrackIDRecords for this camera within the defined time window
    # A TrackIDRecord is within the window if its own [start_time, end_time] overlaps
    # with [analysis_start_time_utc, analysis_end_time_utc]
    query = (
        me.Q(camera=cam_obj_for_filter) &
        (me.Q(start_time__lte=analysis_end_time_utc) & me.Q(end_time__gte=analysis_start_time_utc))
    )
    track_records_for_camera = TrackIDRecord.objects(query)
    
    print(f"Found {len(track_records_for_camera)} TrackIDRecord(s) for {camera_device_name_filter} in the time window.")

    for record in track_records_for_camera:
        # Check if the record has the required zones
        if not (hasattr(record, 'zones') and isinstance(record.zones, dict) and
                inside_zone_label in record.zones and outside_zone_label in record.zones):
            continue

        inside_zone_data = record.zones.get(inside_zone_label)
        outside_zone_data = record.zones.get(outside_zone_label)

        if not (isinstance(inside_zone_data, dict) and inside_zone_data and
                isinstance(outside_zone_data, dict) and outside_zone_data):
            continue # Both zones must have data

        # Parse timestamps
        inside_datetimes = sorted([dt for ts_key in inside_zone_data.keys() if (dt := parse_iso_timestamp_from_zone_key(ts_key)) is not None])
        outside_datetimes = sorted([dt for ts_key in outside_zone_data.keys() if (dt := parse_iso_timestamp_from_zone_key(ts_key)) is not None])

        if not (outside_datetimes and inside_datetimes): # Both must have valid, parsable timestamps
            continue
            
        min_outside = min(outside_datetimes)
        max_outside = max(outside_datetimes)
        min_inside = min(inside_datetimes)
        max_inside = max(inside_datetimes)

        # Prepare details dict (similar to user's `current_record_details_dict`)
        record_details = {
            "db_obj_id": str(record.id),
            "videos": record.videos, # This is List[str] of paths
            "label": record.label,

            "full_db_track_id": record.track_id,
            "record_start_time": record.start_time.isoformat() if record.start_time else "N/A",
            "record_end_time": record.end_time.isoformat() if record.end_time else "N/A",
            # For cropping, we need the raw zone data to get metadata ObjectIds
            "raw_inside_zone_data": inside_zone_data,
            "raw_outside_zone_data": outside_zone_data,
            # Parsed datetimes for potential debugging or further analysis
            "parsed_inside_datetimes_iso": [dt.isoformat() for dt in inside_datetimes],
            "parsed_outside_datetimes_iso": [dt.isoformat() for dt in outside_datetimes]
        }

        # Group by record.track_id (this acts as the 'group_key' or 'track_id_name')
        key_for_filtered_group = record.track_id 

        # Condition for "traveled_inside"
        if min_outside < min_inside and max_outside < max_inside:
            specific_camera_results[camera_device_name_filter]["traveled_inside"][key_for_filtered_group].append(record_details)
        
        # Condition for "traveled_outside"
        if min_outside > min_inside and max_outside > max_inside: # Note: user code had this as `max_outside > max_inside`
            # Corrected logic: min_outside > max_inside implies person was outside, then inside, then went outside again AFTER full inside period
            # Or if min_inside < min_outside and max_inside < max_outside implies inside then outside
            # User's logic: min_outside > min_inside AND max_outside > max_inside: (Entered inside, then exited outside, and remained outside)
            specific_camera_results[camera_device_name_filter]["traveled_outside"][key_for_filtered_group].append(record_details)

# --- Part 3: Other Cameras Logic (Duration Threshold) ---
print("\n--- Part 3: Other Camera Filtering (Duration Threshold) ---")

other_cameras_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# Structure: other_cameras_results[camera_name_str][zone_name_str][track_id_str] = [list of record_details_dict]

DURATION_THRESHOLD_SECONDS = 5.0
print(f"Duration threshold for other cameras: {DURATION_THRESHOLD_SECONDS} seconds")

all_camera_docs = Cameras.objects()
other_camera_docs = [cam_doc for cam_doc in all_camera_docs if cam_doc.device_name not in cameras_to_process_specifically]

for cam_obj_for_filter in other_camera_docs:
    camera_device_name_filter = cam_obj_for_filter.device_name
    print(f"\nProcessing for other camera: {camera_device_name_filter}...")

    query = (
        me.Q(camera=cam_obj_for_filter) &
        (me.Q(start_time__lte=analysis_end_time_utc) & me.Q(end_time__gte=analysis_start_time_utc))
    )
    track_records_for_camera = TrackIDRecord.objects(query)
    print(f"Found {len(track_records_for_camera)} TrackIDRecord(s) for {camera_device_name_filter} in the time window.")

    for record in track_records_for_camera:
        if not (record.start_time and record.end_time and isinstance(record.start_time, datetime.datetime) and isinstance(record.end_time, datetime.datetime)):
            continue
        
        s_time = record.start_time.replace(tzinfo=datetime.timezone.utc) if record.start_time.tzinfo is None else record.start_time
        e_time = record.end_time.replace(tzinfo=datetime.timezone.utc) if record.end_time.tzinfo is None else record.end_time

        if e_time < s_time: continue # Invalid duration

        duration = e_time - s_time
        if duration.total_seconds() <= DURATION_THRESHOLD_SECONDS: # User wants > threshold, so skip if <=
            continue

        # Record meets duration criteria, now categorize by zones
        if hasattr(record, 'zones') and isinstance(record.zones, dict) and record.zones:
            for actual_zone_name, actual_zone_data in record.zones.items():
                if not (isinstance(actual_zone_data, dict) and actual_zone_data): continue # Skip empty/invalid zone data
                if actual_zone_name == "no-zone": continue # Skip "no-zone"

                details_for_this_zone_entry = {
                    "db_obj_id": str(record.id),
                    "videos": record.videos,
                    "label": record.label,

                    "full_db_track_id": record.track_id,
                    "record_start_time_iso": s_time.isoformat(),
                    "record_end_time_iso": e_time.isoformat(),
                    "record_duration_seconds": duration.total_seconds(),
                    "zone_name_of_this_entry": actual_zone_name,
                    # "zone_data_of_this_entry": actual_zone_data # {timestamp_str: OID} dict for this zone
                    "interaction_count_in_zone": len(actual_zone_data)
                }
                other_cameras_results[camera_device_name_filter][actual_zone_name][record.track_id].append(details_for_this_zone_entry)
    
# --- Summary Print for Part 1, 2, 3 (Optional) ---
print("\n--- Filtered Data Summary ---")
for cam_name, results in specific_camera_results.items():
    print(f"\nCamera: {cam_name}")
    if results["traveled_inside"]:
        print(f"  Traveled Inside ({len(results['traveled_inside'])} track_ids):")
        for track_id, rec_list in results["traveled_inside"].items():
             print(f"    Track ID {track_id}: {len(rec_list)} record(s) satisfying criteria")
    if results["traveled_outside"]:
        print(f"  Traveled Outside ({len(results['traveled_outside'])} track_ids):")
        for track_id, rec_list in results["traveled_outside"].items():
             print(f"    Track ID {track_id}: {len(rec_list)} record(s) satisfying criteria")

for cam_name, zone_results in other_cameras_results.items():
    print(f"\nOther Camera: {cam_name}")
    for zone_name, track_id_results in zone_results.items():
        print(f"  Zone: {zone_name} ({len(track_id_results)} track_ids with duration > {DURATION_THRESHOLD_SECONDS}s)")
        # for track_id, rec_list in track_id_results.items():
        #     print(f"    Track ID {track_id}: {len(rec_list)} interaction record(s)")






DEFAULT_REPLACE_DIR_FROM = ""
DEFAULT_REPLACE_DIR_WITH = ""
import os
import subprocess
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm

# Silence OpenCV’s FFmpeg logs
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
except AttributeError:
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

def find_ffmpeg() -> str:
    return shutil.which("ffmpeg") or ""

def remux_copy(src: Path, dst: Path, ffmpeg_bin: str) -> bool:
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(src),
        "-c", "copy",
        "-fflags", "+genpts",
        str(dst)
    ]
    return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def remux_reencode(src: Path, dst: Path, ffmpeg_bin: str) -> bool:
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "copy",
        str(dst)
    ]
    return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


import sys
import os
import contextlib

def extract_and_save_crops_from_video(video_path_str: str, crops_to_extract_list: list):
    # … your remux + VideoCapture setup here …

    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        print(f"    ERROR: cannot open video '{video_path_str}'")
        return 0

    successful = 0
    total = len(crops_to_extract_list)
    print(f"    Processing video: {Path(video_path_str).name} for {total} crops…")

    try:
        for info in tqdm(crops_to_extract_list, desc="Cropping", unit="crop", leave=False):
            frame_idx = info['frame_number']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


            # --- Drop bad frames AND silence FFmpeg’s stderr ---
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
                try:
                    ret, frame = cap.read()
                except cv2.error:
                    continue  # decode error → skip

            if not ret or frame is None:
                continue  # also skip if read() just returned False

            # … your existing padding + bbox logic …
            x1, y1, x2, y2 = map(int, info['bbox'])
            P = 10
            h, w = frame.shape[:2]
            x1p, y1p = max(0, x1 - P), max(0, y1 - P)
            x2p, y2p = min(w, x2 + P), min(h, y2 + P)
            if x2p <= x1p or y2p <= y1p:
                continue

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            out_fp = Path(info['output_filepath'])
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(out_fp), crop):
                successful += 1

    finally:
        cap.release()
        # clean up remux file if any…

    print(f"    INFO: Saved {successful}/{total} crops.")
    return successful


# --- Part 4: Cropping (based on specific_camera_results) ---
print("\n--- Part 4: Cropping ---")

RUN_ID_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = Path(f"./results/track_ids_analysis_crops/run_{RUN_ID_TIMESTAMP}")
base_output_dir.mkdir(parents=True, exist_ok=True)
print(f"Base directory for crops: {base_output_dir.resolve()}")

for camera_name, camera_data in specific_camera_results.items():
    for category_key, track_id_groups in camera_data.items():
        if not track_id_groups:
            continue

        print(f"\n--- Cropping for Camera: {camera_name}, Category: {category_key} ---")
        for full_db_track_id, records in track_id_groups.items():
            print(f"  Processing Track ID: {full_db_track_id} ({len(records)} records)")

            # group crops by video to minimize reopen
            crops_by_video = defaultdict(list)

            for rec in records:
                videos = rec.get('videos', [])
                if not videos:
                    continue
                video_src = videos[0]

                zones = {
                    "outside": rec.get('raw_outside_zone_data', {}),
                    "inside": rec.get('raw_inside_zone_data', {})
                }

                for zone_label, zone_map in zones.items():
                    if not isinstance(zone_map, dict):
                        continue
                    for _, oid_val in zone_map.items():
                        if isinstance(oid_val, ObjectId):
                            oid = oid_val
                        elif isinstance(oid_val, str) and ObjectId.is_valid(oid_val):
                            oid = ObjectId(oid_val)
                        else:
                            continue

                        meta = Metadata.objects(id=oid)\
                                       .only('time_stamp','evidence_frame_number','track_ids_info')\
                                       .first()
                        if not meta:
                            continue

                        info = meta.track_ids_info.get(full_db_track_id)
                        if not info or 'bbox' not in info:
                            continue

                        ts_str = meta.time_stamp.strftime("%Y%m%d%H%M%S%f")[:-3]
                        simple_label = "inside" if "inside" in category_key else "outside"
                        out_dir = base_output_dir / camera_name / full_db_track_id
                        out_dir.mkdir(parents=True, exist_ok=True)

                        fname = f"{full_db_track_id}_{camera_name}_{zone_label}_{ts_str}_{simple_label}.jpg"
                        crops_by_video[video_src].append({
                            "frame_number": meta.evidence_frame_number,
                            "bbox": info['bbox'],
                            "output_filepath": str(out_dir / fname),
                            "full_db_track_id": full_db_track_id
                        })

            for vid_path, crop_list in crops_by_video.items():
                if crop_list:
                    print(f"    For video '{Path(vid_path).name}', extracting {len(crop_list)} crops…")
                    # sort so we don’t jump around wildly
                    crop_list.sort(key=lambda x: x['frame_number'])
                    extract_and_save_crops_from_video(vid_path, crop_list)

print("\n--- All processing and cropping finished ---")
