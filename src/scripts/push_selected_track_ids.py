#!/usr/bin/env python3
"""
Script: bin_and_archive_tracks.py

This script:
1. Uses a default store name to look up a Stores document via the Stores schema.
2. Retrieves that store’s `store_id` and `_id`.
3. Takes an optional start_time and end_time (ISO format, treated as Asia/Kolkata local time if no timezone).
   If not provided, defaults to:
       start_time = "2025-06-01T11:00:00"
       end_time   = "2025-06-01T12:00:00"
4. Converts them to UTC.
5. Splits the interval into consecutive 1-hour bins.
6. For each bin:
   a. Filters TrackIDRecords in MongoDB overlapping that bin.
   b. Archives the matching crops into a folder named:
        "./results/track_id_archive_<bin_start>-<bin_end>"
   c. Creates a `.tar.gz` of that folder.
   d. Uploads the archive to Azure Blob Storage.
   e. Records metadata (start_time, end_time, duration, blob_url, status) under a document in the
      "track_id_archives" collection. Each document is keyed by the local date (Asia/Kolkata),
      and stores `store_id` and `store_doc_id` from the looked-up store document, plus a `pushes_dict`
      mapping each bin’s run to its metadata.
7. Writes detailed logs (INFO and ERROR) to both the console and a daily log file at:
       ./logs/track_id_archives/logs_<YYYY-MM-DD>.logs

Usage:
    python bin_and_archive_tracks.py
       (uses default 2025-06-01T11:00:00 → 2025-06-01T12:00:00)

    Or override:
    python bin_and_archive_tracks.py \
        --start_time "2025-06-01T08:00:00" \
        --end_time   "2025-06-01T12:30:00"
"""

import os
import sys
import argparse
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import shutil
import tarfile
import logging
from tqdm import tqdm

import dotenv
from dotenv import load_dotenv

import mongoengine as me

from azure.storage.blob import BlobServiceClient

# ──────────────────────────────────────────────────────────────────────────────
# Configuration: Default Store Name & Default Time Window
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_STORE_NAME = "Reliance-Smart-Dombivali"
# DEFAULT_START_TIME = "2025-06-02T12:00:00"
# DEFAULT_END_TIME   = "2025-06-02T14:00:00"

# DEFAULT_START_TIME = ["2025-06-02T16:00:00", "2025-06-02T20:00:00"]
# DEFAULT_END_TIME   = ["2025-06-02T17:00:00", "2025-06-02T21:00:00"]

DEFAULT_START_TIME = ["2025-06-30T11:00:00"]
DEFAULT_END_TIME   = ["2025-06-30T12:00:00"]

# ──────────────────────────────────────────────────────────────────────────────
# MongoEngine Document for track_id_archives
# ──────────────────────────────────────────────────────────────────────────────


# Try to import your existing pipeline function:
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(SCRIPT_DIR))

class TrackIDArchive(me.Document):
    """
    Stores archive metadata for a given local date (Asia/Kolkata).
    Each document's primary key is the date string "YYYY-MM-DD".
    """
    date = me.StringField(primary_key=True)       # e.g. "2025-06-01"
    store_id = me.StringField(required=True)      # e.g. "001"
    store_doc_id = me.StringField(required=True)  # e.g. "68075b1bc517c730d3a21638"
    pushes_dict = me.DictField(default={})        # { "run_<bin_start>-<bin_end>": {start_time, end_time, duration, blob_url, status}, ... }

    meta = {
        'collection': 'track_id_archives'
    }

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Parse ISO string to UTC-aware datetime (interpreting naive as Asia/Kolkata)
# ──────────────────────────────────────────────────────────────────────────────
def parse_iso_to_utc(iso_str: str) -> datetime.datetime:
    """
    Interpret iso_str (e.g. "2025-06-01T10:00:00") as Asia/Kolkata local time
    (if no timezone is present), then convert to UTC. Returns a timezone-aware UTC datetime.
    """
    dt = datetime.datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    dt_utc = dt.astimezone(datetime.timezone.utc)
    return dt_utc

# ──────────────────────────────────────────────────────────────────────────────
# Generate hourly bins between start_utc and end_utc
# ──────────────────────────────────────────────────────────────────────────────
def generate_hourly_bins(start_utc: datetime.datetime, end_utc: datetime.datetime):
    """
    Yields tuples of (bin_start_utc, bin_end_utc) for each 1-hour bin,
    covering [start_utc, end_utc). The final bin_end is <= end_utc.
    """
    current = start_utc
    one_hour = datetime.timedelta(hours=1)
    while current < end_utc:
        bin_start = current
        bin_end = min(current + one_hour, end_utc)
        yield bin_start, bin_end
        current = bin_end

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Parse ISO timestamp from zone key (copied from original filtering code)
# ──────────────────────────────────────────────────────────────────────────────
def parse_iso_timestamp_from_zone_key(ts_str):
    """
    Helper function to parse ISO timestamp strings from zone data keys.
    Returns a naive datetime (no timezone).
    """
    if not ts_str:
        return None
    try:
        if ts_str.endswith('+0000'):
            ts_str = ts_str[:-5] + '+00:00'
        elif ts_str.endswith('-0000'):
            ts_str = ts_str[:-5] + '-00:00'
        return datetime.datetime.fromisoformat(ts_str)
    except ValueError:
        try:
            return datetime.datetime.fromisoformat(ts_str.split('+')[0].split('-')[0])
        except ValueError:
            print(f"Warning: Could not parse zone timestamp string: '{ts_str}'")
            return None

# ──────────────────────────────────────────────────────────────────────────────
# Filtering logic: Given a UTC interval, retrieve specific and other camera results
# ──────────────────────────────────────────────────────────────────────────────
def filter_records_for_time_window(analysis_start_time_utc, analysis_end_time_utc):
    """
    Implements the filtering logic (Parts 1, 2, 3 from the provided notebook),
    returning two dicts: specific_camera_results and other_cameras_results.
    """
    from src.database.schemas.sentinel_poc_schema import TrackIDRecord, Cameras  # noqa: E402

    # Part 1 & 2: Specific Camera Logic for "Camera-018" and "Camera-016"
    specific_camera_results = {
        "Camera-010": {"traveled_inside": dict(), "traveled_outside": dict()}
    }
    cameras_to_process_specifically = ["Camera-018", "Camera-016"]

    for camera_device_name_filter in cameras_to_process_specifically:
        # Determine zone labels
        if camera_device_name_filter == "Camera-010":
            inside_zone_label = "inside (exit)"
            outside_zone_label = "outside (entry)"
        else:
            continue

        cam_obj_for_filter = Cameras.objects(device_name=camera_device_name_filter).first()
        if not cam_obj_for_filter:
            continue

        query = (
            me.Q(camera=cam_obj_for_filter) &
            (me.Q(start_time__lte=analysis_end_time_utc) & me.Q(end_time__gte=analysis_start_time_utc))
        )
        track_records_for_camera = TrackIDRecord.objects(query)

        for record in track_records_for_camera:
            if not (hasattr(record, 'zones') and isinstance(record.zones, dict) and
                    inside_zone_label in record.zones and outside_zone_label in record.zones):
                continue

            inside_zone_data = record.zones.get(inside_zone_label)
            outside_zone_data = record.zones.get(outside_zone_label)
            if not (isinstance(inside_zone_data, dict) and inside_zone_data and
                    isinstance(outside_zone_data, dict) and outside_zone_data):
                continue

            inside_datetimes = sorted(
                dt for ts_key in inside_zone_data.keys()
                if (dt := parse_iso_timestamp_from_zone_key(ts_key)) is not None
            )
            outside_datetimes = sorted(
                dt for ts_key in outside_zone_data.keys()
                if (dt := parse_iso_timestamp_from_zone_key(ts_key)) is not None
            )
            if not (outside_datetimes and inside_datetimes):
                continue

            min_outside = min(outside_datetimes)
            max_outside = max(outside_datetimes)
            min_inside = min(inside_datetimes)
            max_inside = max(inside_datetimes)

            record_details = {
                "db_obj_id": str(record.id),
                "videos": record.videos,
                "label": record.label,
                "track_id_path_lists": record.track_id_path_lists,
                "full_db_track_id": record.track_id,
                "record_start_time": record.start_time.isoformat() if record.start_time else "N/A",
                "record_end_time": record.end_time.isoformat() if record.end_time else "N/A",
                "raw_inside_zone_data": inside_zone_data,
                "raw_outside_zone_data": outside_zone_data,
                "parsed_inside_datetimes_iso": [dt.isoformat() for dt in inside_datetimes],
                "parsed_outside_datetimes_iso": [dt.isoformat() for dt in outside_datetimes],
            }
            key_for_filtered_group = record.track_id

            # traveled_inside: outside first, then inside
            if min_outside < min_inside and max_outside < max_inside:
                specific_camera_results[camera_device_name_filter]["traveled_inside"]\
                    .setdefault(key_for_filtered_group, []).append(record_details)

            # traveled_outside: inside first, then outside
            if min_outside > min_inside and max_outside > max_inside:
                specific_camera_results[camera_device_name_filter]["traveled_outside"]\
                    .setdefault(key_for_filtered_group, []).append(record_details)

    # Part 3: Other Cameras (duration threshold)
    other_cameras_results = dict()  # { camera_name: { zone_name: { track_id: [details, ...] } } }
    DURATION_THRESHOLD_SECONDS = 5.0

    all_camera_docs = Cameras.objects()
    other_camera_docs = [cam for cam in all_camera_docs if cam.device_name not in cameras_to_process_specifically]

    for cam_obj_for_filter in other_camera_docs:
        camera_device_name_filter = cam_obj_for_filter.device_name
        query = (
            me.Q(camera=cam_obj_for_filter) &
            (me.Q(start_time__lte=analysis_end_time_utc) & me.Q(end_time__gte=analysis_start_time_utc))
        )
        track_records_for_camera = TrackIDRecord.objects(query)

        for record in track_records_for_camera:
            if not (record.start_time and record.end_time and
                    isinstance(record.start_time, datetime.datetime) and isinstance(record.end_time, datetime.datetime)):
                continue

            s_time = record.start_time.replace(tzinfo=datetime.timezone.utc) if record.start_time.tzinfo is None else record.start_time
            e_time = record.end_time.replace(tzinfo=datetime.timezone.utc) if record.end_time.tzinfo is None else record.end_time
            if e_time < s_time:
                continue

            duration = (e_time - s_time).total_seconds()
            if duration <= DURATION_THRESHOLD_SECONDS:
                continue

            if hasattr(record, 'zones') and isinstance(record.zones, dict) and record.zones:
                for actual_zone_name, actual_zone_data in record.zones.items():
                    if not (isinstance(actual_zone_data, dict) and actual_zone_data):
                        continue
                    if actual_zone_name == "no-zone":
                        continue

                    details_for_this_zone_entry = {
                        "db_obj_id": str(record.id),
                        "videos": record.videos,
                        "label": record.label,
                        "track_id_path_lists": record.track_id_path_lists,
                        "full_db_track_id": record.track_id,
                        "record_start_time_iso": s_time.isoformat(),
                        "record_end_time_iso": e_time.isoformat(),
                        "record_duration_seconds": duration,
                        "zone_name_of_this_entry": actual_zone_name,
                        "interaction_count_in_zone": len(actual_zone_data),
                    }
                    other_cameras_results.setdefault(camera_device_name_filter, {}) \
                                         .setdefault(actual_zone_name, {}) \
                                         .setdefault(record.track_id, []).append(details_for_this_zone_entry)

    return specific_camera_results, other_cameras_results

# ──────────────────────────────────────────────────────────────────────────────
# Archive cropping results to local folder and copy to "./results/..._binstart-binend"
# ──────────────────────────────────────────────────────────────────────────────
def archive_tracks(specific_camera_results, other_cameras_results, archive_root: str = "./results_archive"):
    """
    For each camera/bin grouping in specific_camera_results and other_cameras_results,
    copy all crop files into `archive_root` preserving relative paths.
    `archive_root` should be something like "./results/track_id_archive_20250601T080000Z-20250601T090000Z"
    """
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    exclude_str = "no-zone"

    def copy_paths(nested_path_lists, desc):
        """
        nested_path_lists: List[List[List[str]]]
        desc:               str for tqdm description
        """
        flat_paths = []
        for group in nested_path_lists:
            for rec_paths in group:
                for src_path in rec_paths:
                    if exclude_str in src_path:
                        continue
                    flat_paths.append(src_path)

        for src_path in tqdm(flat_paths, desc=desc, unit="file"):
            src = Path(src_path)
            if not src.is_file():
                tqdm.write(f"⚠️  Missing: {src}")
                continue
            try:
                rel = src.relative_to(".")
            except Exception:
                rel = Path(src.name)
            dest = archive_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # 1) Specific cameras
    for cam_name, directions in tqdm(specific_camera_results.items(), desc="Specific Cameras", unit="camera"):
        if cam_name == "Camera-018":
            direction = "traveled_inside"
            tracks = directions.get(direction, {})
            all_lists = []
            for rec_list in tracks.values():
                for rec in rec_list:
                    all_lists.append(rec.get("track_id_path_lists", []))
            if all_lists:
                copy_paths(all_lists, desc=f"{cam_name} → {direction}")

        elif cam_name == "Camera-016":
            direction = "traveled_outside"
            tracks = directions.get(direction, {})
            all_lists = []
            for rec_list in tracks.values():
                for rec in rec_list:
                    all_lists.append(rec.get("track_id_path_lists", []))
            if all_lists:
                copy_paths(all_lists, desc=f"{cam_name} → {direction}")

    # 2) Other cameras
    for cam_name, zones in tqdm(other_cameras_results.items(), desc="Other Cameras", unit="camera"):
        all_lists = []
        for recs_by_track in zones.values():
            for rec_list in recs_by_track.values():
                for rec in rec_list:
                    all_lists.append(rec.get("track_id_path_lists", []))
        if all_lists:
            copy_paths(all_lists, desc=cam_name)

# ──────────────────────────────────────────────────────────────────────────────
# Create a .tar.gz archive of a folder
# ──────────────────────────────────────────────────────────────────────────────
def create_tar_gz_archive(folder_path) -> Path | None:
    """
    Given a folder_path (string or Path), create a .tar.gz archive alongside it.
    Returns the Path to the .tar.gz file, or None if the folder was empty or an error occurred.
    """
    folder_path = Path(folder_path)
    if not any(folder_path.rglob('*')):
        logging.info(f"Folder {folder_path} is empty. No archive will be created.")
        return None

    archive_path = folder_path.with_suffix('.tar.gz')
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)
        logging.info(f"Created archive: {archive_path}")
        return archive_path
    except Exception as e:
        logging.error(f"Failed to create archive {archive_path}: {e}")
        return None
# ──────────────────────────────────────────────────────────────────────────────
# Upload a file to Azure Blob Storage; returns blob URL or None on failure
# ──────────────────────────────────────────────────────────────────────────────
def upload_to_azure_blob(file_path: Path, connection_string: str, container_name: str) -> str | None:
    """
    Uploads file_path to Azure Blob Storage under `container_name`.
    Returns the blob URL or None if upload fails.
    """
    if not connection_string or not container_name:
        logging.error("Azure connection string or container name is not configured.")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Suppress verbose HTTP logging from the Azure SDK:
    # Any logger under "azure" or the HTTP logging policy is bumped up to WARNING.
    # ──────────────────────────────────────────────────────────────────────────
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('azure.storage').setLevel(logging.WARNING)
    logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)


    try:
        # 2) Disable HTTP logging in the client itself
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string,
            logging_enable=False
        )
        container_client = blob_service_client.get_container_client(container_name)

        # If the container does not exist, attempt to create it
        try:
            container_client.create_container()
        except Exception as e:
            # Ignore “ContainerAlreadyExists” but warn on anything else
            if "ContainerAlreadyExists" not in str(e):
                logging.warning(f"Could not create container: {e}")

        blob_name = file_path.name
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = blob_client.url
        logging.info(f"Uploaded {file_path} to Azure Blob as '{blob_name}'")
        return blob_url

    except Exception as e:
        logging.error(f"Failed to upload {file_path} to Azure Blob Storage: {e}")
        return None

def parse_input_times(start_iso, end_iso, args=None, logger=None):
    # Lookup default store document via Stores schema

        # Load environment variables
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    # DB_NAME = os.getenv("DB_NAME")
    AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    try:
        from src.database.schemas.sentinel_poc_schema import Stores  # noqa: E402

        store_doc = Stores.objects(name=DEFAULT_STORE_NAME).first()
        if not store_doc:
            raise RuntimeError(f"Store with name '{DEFAULT_STORE_NAME}' not found in Stores collection.")
        store_id_value = store_doc.store_id
        store_doc_id_value = str(store_doc.id)
        logger.info(f"Found store: {DEFAULT_STORE_NAME} (store_id={store_id_value}, _id={store_doc_id_value})")
    except Exception as e:
        logger.error(f"Error fetching store document: {e}")
        sys.exit(1)

    # Parse input times as UTC
    try:
        start_utc = parse_iso_to_utc(start_iso)
        end_utc = parse_iso_to_utc(end_iso)
        logger.info(f"Analysis window (UTC): {start_utc.isoformat()} → {end_utc.isoformat()}")
    except Exception as e:
        logger.error(f"Error parsing input times: {e}")
        sys.exit(1)

    if end_utc <= start_utc:
        logger.error("end_time must be after start_time.")
        sys.exit(1)

    # Iterate hourly bins
    for bin_start_utc, bin_end_utc in generate_hourly_bins(start_utc, end_utc):
        # Format bin labels as YYYYMMDDTHHMMSSZ for filenames and keys
        bin_start_label = bin_start_utc.strftime("%Y%m%dT%H%M%SZ")
        bin_end_label = bin_end_utc.strftime("%Y%m%dT%H%M%SZ")
        archive_root = f"./results/track_id_archive/track_id_archive_{bin_start_label}-{bin_end_label}"
        run_key = f"run_{bin_start_label}-{bin_end_label}"

        # 5) Compute local date (Asia/Kolkata) for this bin_start
        local_dt = bin_start_utc.astimezone(ZoneInfo("Asia/Kolkata"))
        local_date_str = local_dt.date().isoformat()  # e.g. "2025-06-01"

        try:
            archive_doc = TrackIDArchive.objects(date=local_date_str).first()
            if archive_doc:
                if run_key in archive_doc.pushes_dict and archive_doc.pushes_dict[run_key]["status"] == "done" and \
                    len(archive_doc.pushes_dict[run_key]["blob_url"]) > 2:
                    continue

        except Exception as e:
            logger.error(f"Error getting TrackIDArchive document for date {local_date_str}: {e}")




        # try:
        #     archive_doc = TrackIDArchive.objects(date=local_date_str).first()
        #     if archive_doc:
        #         archive_doc.pushes_dict[run_key] = run_entry
        #         archive_doc.save()
        #         logger.info(f"Updated track_id_archives for date {local_date_str}, added {run_key}")


        archive_dir = Path(archive_root).parent

        if os.path.exists(archive_dir):
            shutil.rmtree(archive_dir)
            logger.info(f"Deleted existing archive directory: {archive_dir}")
            

        logger.info(f"Processing bin: {bin_start_label} → {bin_end_label}")

        # 1) Filter records
        try:
            specific_camera_results, other_cameras_results = filter_records_for_time_window(
                bin_start_utc, bin_end_utc
            )
            logger.info(f"Filtered records for bin {bin_start_label}-{bin_end_label}")
        except Exception as e:
            logger.error(f"Error during filtering for bin {bin_start_label}-{bin_end_label}: {e}")
            run_status = f"error (filtering): {e}"
            blob_url = None
        else:
            # 2) Archive crops to local folder
            try:
                logger.info(f"Archiving crops to {archive_root}")
                archive_tracks(specific_camera_results, other_cameras_results, archive_root=archive_root)
                logger.info(f"Archived crops to {archive_root}")
            except Exception as e:
                logger.error(f"Error during archiving files for bin {bin_start_label}-{bin_end_label}: {e}")
                run_status = f"error (archiving): {e}"
                blob_url = None
            else:
                # 3) Create tar.gz of archive_root
                archive_path = create_tar_gz_archive(archive_root)
                if archive_path:
                    logger.info(f"Created archive: {archive_path}")
                    # 4) Upload to Azure Blob
                    try:
                        blob_url = upload_to_azure_blob(archive_path, AZURE_CONN_STR, AZURE_CONTAINER)
                        if blob_url:
                            run_status = "done"
                            logger.info(f"Uploaded archive to Azure: {blob_url}")
                        else:
                            run_status = "error (upload_failed)"
                            logger.error("Upload to Azure failed (no URL returned).")
                    except Exception as e:
                        logger.error(f"Error uploading to Azure for bin {bin_start_label}-{bin_end_label}: {e}")
                        run_status = f"error (upload): {e}"
                        blob_url = None
                else:
                    run_status = "error (archive_creation_failed)"
                    blob_url = None

        # 5) Compute local date (Asia/Kolkata) for this bin_start
        local_dt = bin_start_utc.astimezone(ZoneInfo("Asia/Kolkata"))
        local_date_str = local_dt.date().isoformat()  # e.g. "2025-06-01"

        # 6) Prepare run key and entry
        run_entry = {
            "start_time": bin_start_utc.isoformat(),
            "end_time": bin_end_utc.isoformat(),
            "duration_seconds": int((bin_end_utc - bin_start_utc).total_seconds()),
            "blob_url": blob_url,
            "status": run_status,
        }

        # 7) Upsert document in track_id_archives
        try:
            archive_doc = TrackIDArchive.objects(date=local_date_str).first()
            if archive_doc:
                archive_doc.pushes_dict[run_key] = run_entry
                archive_doc.save()
                logger.info(f"Updated track_id_archives for date {local_date_str}, added {run_key}")
            else:
                new_doc = TrackIDArchive(
                    date=local_date_str,
                    store_id=store_id_value,
                    store_doc_id=store_doc_id_value,
                    pushes_dict={run_key: run_entry}
                )
                new_doc.save()
                logger.info(f"Created new track_id_archives doc for date {local_date_str}, run {run_key}")
        except Exception as e:
            logger.error(f"Error updating Mongo document for date {local_date_str}: {e}")


    logger.info("All bins processed. Exiting.")

# ──────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Bin and archive TrackIDRecords into hourly archives and push to Azure Blob."
    )
    parser.add_argument(
        "--start_time", "-s",
        help="ISO datetime (e.g. '2025-06-01T08:00:00') treated as Asia/Kolkata if no timezone."
    )
    parser.add_argument(
        "--end_time", "-e",
        help="ISO datetime (e.g. '2025-06-01T12:30:00') treated as Asia/Kolkata if no timezone."
    )
    args = parser.parse_args()


    # Load environment variables
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    # DB_NAME = os.getenv("DB_NAME")
    AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

    if not (MONGODB_URI):
        print("Error: MONGODB_URI and DB_NAME must be set in environment.")
        sys.exit(1)

    # Connect to MongoDB
    try:
        me.disconnect_all()
        me.connect(host=MONGODB_URI, UuidRepresentation='standard', alias='default')
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        sys.exit(1)

    # Set up logging directory and file (daily)
    try:
        # Get the current local datetime for "Asia/Kolkata"
        local_now = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))

        # Store date and time as a datetime string using isoformat()
        # This will typically produce a string like "2025-06-04T14:01:22.123456+05:30"
        log_date_str = local_now.isoformat()
        # log_date_str = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
        log_dir = Path("./logs/02-track_id_archives")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"logs_{log_date_str}.logs"

        # Configure the root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (appends to today's log file)
        fh = logging.FileHandler(log_file_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info(f"Logging initialized. Logs will be written to {log_file_path}")
    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1)


        # Determine start/end or defaults
    start_iso_input = args.start_time or DEFAULT_START_TIME
    end_iso_input = args.end_time or DEFAULT_END_TIME

    if isinstance(start_iso_input, list) and isinstance(end_iso_input, list):
        for start_iso, end_iso in zip(start_iso_input, end_iso_input):
            parse_input_times(start_iso, end_iso, args, logger)

    elif isinstance(start_iso_input, str) and isinstance(end_iso_input, str):
        start_iso = start_iso_input
        end_iso = end_iso_input
        parse_input_times(start_iso, end_iso, args, logger)
    else:
        logger.error("Invalid input times. Please provide start and end times as ISO strings or lists.")
        sys.exit(1)



if __name__ == "__main__":
    main()
