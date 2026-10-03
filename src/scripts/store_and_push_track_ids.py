import mongoengine as me
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import cv2
import sys
import tarfile
import traceback
# import datetime as dt # dt alias no longer strictly needed if all defaults use timezone.utc
from multiprocessing import Pool, cpu_count
from mongoengine import (Document, IntField, DateTimeField, StringField, 
                         DictField, FloatField, ListField, ReferenceField, 
                         BooleanField, LongField) # Keep these for locally defined schemas
from collections import defaultdict
from azure.storage.blob import BlobServiceClient
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import pytz # For timezone conversions
import json # Add this import at the top of your script

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
MONGODB_URI = os.getenv("MONGODB_URI")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()


# === ADD THIS SECTION TO QUIET AZURE SDK LOGGERS ===
azure_logger_names = [
    "azure.core.pipeline.policies. універсальногоHttpPolicy",
    "azure.core.pipeline.policies",
    "azure.storage.blob",
    "azure" # A general catch-all for other Azure SDK loggers
]
for logger_name in azure_logger_names:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
# === END OF SECTION TO QUIET AZURE SDK LOGGERS ===


# Add src path to sys.path
src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))

from src.database.schemas.sentinel_poc_schema import (
    Cameras, Stores, Zone, CameraZoneMapping, Pipeline, Metadata
)


# === Schemas defined locally (specific to this script's logic) ===
class TrackIDMetadata(Document):
    meta = {'collection': 'track_id_metadata'} # Ensure this collection name is correct
    track_id = StringField(required=True)
    device = StringField(required=True)
    attended = IntField() # Assuming these fields exist
    dwell_time = LongField(required=True) # Assuming these fields exist
    start_time = DateTimeField(required=True)
    end_time = DateTimeField(required=True)
    gender = StringField()
    label = StringField()
    zone = StringField(required=True)
    interacted = IntField() # Assuming these fields exist
    interacted_list = ListField(StringField(), default=[]) # Assuming these fields exist
    # Add any other fields from TrackIDMetadata that are actually used by the script implicitly



class TrackIdCropsRun(me.Document): # MODIFIED SCHEMA
    meta = {'collection': 'track_id_crops_runs'}
    run_start_timestamp = me.DateTimeField(required=True)
    run_end_timestamp = me.DateTimeField(null=True)
    processing_duration_seconds = me.FloatField(null=True)
    query_window_start_utc = me.DateTimeField(null=True)
    query_window_end_utc = me.DateTimeField(null=True)
    query_end_time_ist_str = me.StringField()
    query_duration_minutes = me.IntField()
    archive_blob_url = me.StringField(null=True)
    
    # MODIFIED to store summary info, not all paths
    processed_track_ids_summary = me.DictField( # Key: track_id (string)
        field=me.DictField(
            fields={
                "track_metadata_start_time_utc": me.DateTimeField(),
                "track_metadata_end_time_utc": me.DateTimeField(),
                "device": me.StringField(),
                "zone": me.StringField(),
                "crop_count": me.IntField(default=0) # Stores count of crops for this track
            }
        ),
        default={}
    )
    # manifest_filename = me.StringField(default="manifest.json") # Optionally store manifest filename

    status = me.StringField(default="pending")
    error_message = me.StringField(null=True)
    total_track_ids_found = me.IntField(default=0)
    total_metadata_entries_processed = me.IntField(default=0)
    total_crops_generated = me.IntField(default=0)


# --- Utility Functions (upload_to_azure_blob, create_tar_gz_archive, extract_and_save_crops_from_video) ---
# These functions remain the same. For brevity, they are not repeated here.
def upload_to_azure_blob(file_path: Path, connection_string: str, container_name: str) -> str | None:
    if not connection_string or not container_name:
        logger.error("Azure connection string or container name is not configured. Skipping upload.")
        return None
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
        except Exception as e:
            if "ContainerAlreadyExists" not in str(e):
                 logger.warning(f"Could not create container (it might already exist or an error occurred): {e}")

        blob_name = file_path.name
        blob_client = container_client.get_blob_client(blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        blob_url = blob_client.url
        logger.info(f"Uploaded {file_path} to Azure Blob Storage container '{container_name}' as blob '{blob_name}'")
        return blob_url
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to Azure Blob Storage: {e}")
        return None

def create_tar_gz_archive(folder_path: Path) -> Path | None:
    is_truly_empty = True
    if any(folder_path.iterdir()):
        is_truly_empty = False
        all_dirs_empty = True
        has_files = False
        for item in folder_path.iterdir():
            if item.is_file():
                has_files = True
                all_dirs_empty = False
                break
            if item.is_dir():
                if any(item.iterdir()):
                    all_dirs_empty = False
                    break
        if not has_files and all_dirs_empty and not any(f for f in folder_path.iterdir() if f.is_file()):
             is_truly_empty = True
        else:
            is_truly_empty = False
    if is_truly_empty:
        logger.info(f"Folder {folder_path} is effectively empty. No archive will be created.")
        return None
    archive_path = folder_path.with_suffix('.tar.gz')
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)
        logger.info(f"Created archive: {archive_path}")
        return archive_path
    except Exception as e:
        logger.error(f"Failed to create archive {archive_path}: {e}")
        return None

def extract_and_save_crops_from_video(args_tuple):
    video_path_str, crops_list_for_this_video = args_tuple
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path_str}")
        return 0
    crops_saved_count = 0
    crops_grouped_by_frame_no = defaultdict(list)
    for job in crops_list_for_this_video:
        crops_grouped_by_frame_no[job['frame_number']].append(job)
    sorted_unique_frame_numbers = sorted(crops_grouped_by_frame_no.keys())
    for frame_no_to_read in sorted_unique_frame_numbers:
        all_crops_for_this_frame_exist = True
        for job_in_frame in crops_grouped_by_frame_no[frame_no_to_read]:
            if not os.path.exists(job_in_frame['output_path']):
                all_crops_for_this_frame_exist = False
                break
        if all_crops_for_this_frame_exist:
            logger.debug(f"All crops for frame {frame_no_to_read} in {video_path_str} already exist. Skipping frame read.")
            crops_saved_count += len(crops_grouped_by_frame_no[frame_no_to_read])
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_read)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {frame_no_to_read} from {video_path_str}. Skipping all crops for this frame.")
            continue
        for job in crops_grouped_by_frame_no[frame_no_to_read]:
            if os.path.exists(job['output_path']):
                logger.debug(f"Crop {job['output_path']} (frame {frame_no_to_read}) already exists. Skipping.")
                crops_saved_count += 1
                continue
            bbox = job['bbox']
            out_path = Path(job['output_path'])
            if frame is None or not hasattr(frame, 'shape'):
                logger.warning(f"Invalid frame object for frame {frame_no_to_read} in {video_path_str}.")
                break
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            x1p = max(0, int(x1) - 10); y1p = max(0, int(y1) - 10)
            x2p = min(w, int(x2) + 10); y2p = min(h, int(y2) + 10)
            if not (x1p < x2p and y1p < y2p):
                logger.warning(f"Invalid bounding box for crop {out_path} in {video_path_str}, frame {frame_no_to_read}.")
                continue
            crop_img = frame[y1p:y2p, x1p:x2p]
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), crop_img)
                crops_saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save crop {out_path} (frame {frame_no_to_read}): {e}")
    cap.release()
    return crops_saved_count

# --- Core Logic Functions ---
def get_track_ids_in_window(end_time_ist_str: str | None, duration_minutes: int):
    ist_tz = pytz.timezone('Asia/Kolkata')
    if end_time_ist_str:
        try:
            end_time_ist = ist_tz.localize(datetime.strptime(end_time_ist_str, "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            logger.error(f"Invalid end_time format: {end_time_ist_str}. Please use YYYY-MM-DDTHH:MM:SS.")
            return [], None, None, "Invalid end_time format"
    else:
        end_time_ist = datetime.now(ist_tz)
        logger.info(f"No end_time provided, using current IST time: {end_time_ist.strftime('%Y-%m-%dT%H:%M:%S')}")

    end_time_utc = end_time_ist.astimezone(timezone.utc)
    start_time_utc = end_time_utc - timedelta(minutes=duration_minutes)
    logger.info(f"Querying TrackIDMetadata for tracks starting between UTC: {start_time_utc} and {end_time_utc}")

    try:
        track_id_docs = list(TrackIDMetadata.objects(start_time__gte=start_time_utc, start_time__lt=end_time_utc))
        if track_id_docs:
            device_counts = {}
            for doc in track_id_docs:
                device_counts[doc.device] = device_counts.get(doc.device, 0) + 1
            for device, count in device_counts.items():
                logger.info(f"Device: {device}, Count: {count}")

        
        logger.info(f"Found {len(track_id_docs)} TrackIDMetadata documents in the time window.")
    except me.errors.MongoEngineException as e:
        logger.error(f"Error querying TrackIDMetadata: {e}")
        return [], start_time_utc, end_time_utc, str(e)
    return track_id_docs, start_time_utc, end_time_utc, None


def get_metadata_for_tracks(track_id_docs: list[TrackIDMetadata]): # MODIFIED
    if not track_id_docs:
        logger.info("No TrackIDMetadata documents to process for metadata.")
        return {}, None
        
    all_track_ids_str = list(set([str(t.track_id) for t in track_id_docs]))
    if not all_track_ids_str:
        logger.info("No unique track IDs found from TrackIDMetadata documents.")
        return {}, None

    # Determine the broad time range for Metadata query
    min_start_time = min(t.start_time for t in track_id_docs if t.start_time)
    max_end_time = max(t.end_time for t in track_id_docs if t.end_time)
    logger.info(f"Querying Metadata for {len(all_track_ids_str)} unique track IDs (sample: {all_track_ids_str[:3]}), spanning UTC: {min_start_time} to {max_end_time}")
    # time.sleep(2)
    try:
        metas = list(Metadata.objects(time_stamp__gte=min_start_time, time_stamp__lte=max_end_time))
        logger.info(f"Fetched {len(metas)} Metadata documents from the broad time range.")
    except me.errors.MongoEngineException as e:
        logger.error(f"Error querying Metadata: {e}")
        return {}, str(e)
    except Exception as e: # Catch if Metadata schema is not correctly loaded
        logger.error(f"Generic error querying Metadata (check schema import): {e}")
        return {}, str(e)

    # This will be the final result, as the 5-per-second filter is removed.
    final_trackid_to_metadocs = defaultdict(list)
    
    # To count unique meta_docs that have at least one match to any of our needed track IDs
    meta_docs_with_any_association_ids = set()

    for meta_doc in tqdm(metas, desc="Associating Metadata with Tracks", unit="meta_doc", disable=not metas):
        if not meta_doc.track_ids_info:
            continue

        
        
        doc_track_ids_present_in_meta = meta_doc.track_ids_info.keys()
        found_match_for_this_meta_doc = False
        

        # print("doc_track_ids_present_in_meta", doc_track_ids_present_in_meta)
        # Iterate through the track IDs we are looking for
        for tid_str_we_need in doc_track_ids_present_in_meta: 
            if tid_str_we_need in all_track_ids_str:
                final_trackid_to_metadocs[tid_str_we_need].append(meta_doc)
                found_match_for_this_meta_doc = True
        
        if found_match_for_this_meta_doc:
            # Add meta_doc's ID (or a unique representation) to the set
            # Using object's id() is not stable if objects are recreated.
            # If meta_doc has a database ID field (e.g., meta_doc.id), that's best.
            if hasattr(meta_doc, 'id') and meta_doc.id is not None:
                 meta_docs_with_any_association_ids.add(meta_doc.id)
            else:
                # Fallback if no .id, less reliable for uniqueness if objects can be identical but different instances
                # For MongoEngine documents, .id should exist after fetching from DB.
                 meta_docs_with_any_association_ids.add(id(meta_doc))


    associated_meta_doc_count = len(meta_docs_with_any_association_ids)
    total_associations = sum(len(v) for v in final_trackid_to_metadocs.values())

    logger.info(f"Association complete: {total_associations} total pairings made. {associated_meta_doc_count} unique Metadata documents had at least one relevant track ID.")
    logger.info(f"{len(final_trackid_to_metadocs)} TrackIDs have associated Metadata entries.")
    
    # The 5-per-second filtering loop is now removed.
    # 'final_trackid_to_metadocs' contains all associated metadata documents for each track ID.

    total_entries_to_process = sum(len(docs) for docs in final_trackid_to_metadocs.values())
    logger.info(f"Total Metadata entries to process for crops (no 5/sec filter): {total_entries_to_process}")
    
    return final_trackid_to_metadocs, None


def prepare_crop_jobs_and_info(track_id_docs: list[TrackIDMetadata], 
                               trackid_to_all_metadocs: dict, 
                               base_dir: Path):
    crops_by_video_path_key = defaultdict(list)
    
    # This will hold the full details for the JSON manifest
    full_manifest_data = {} 
    # This will hold summary details for MongoDB
    summary_track_details_for_db = {}
    
    total_crops_planned = 0

    logger.info(f"Preparing crop jobs and manifest data. Base directory: {base_dir}")

    for track_doc in tqdm(track_id_docs, desc="Preparing Crop Jobs & Manifest", unit="track_doc", disable=not track_id_docs):
        if not hasattr(track_doc, 'track_id') or track_doc.track_id is None:
            logger.warning("DEBUG: Skipping a TrackIDMetadata doc with missing or None track_id.")
            continue
        track_id_str = track_doc.track_id

        zone_name = getattr(track_doc, 'zone', 'unknown_zone')
        device_name = getattr(track_doc, 'device', 'unknown_device')

        # For full_manifest_data
        current_track_full_info = {
            "track_metadata_start_time_utc": track_doc.start_time.isoformat() if track_doc.start_time else None,
            "track_metadata_end_time_utc": track_doc.end_time.isoformat() if track_doc.end_time else None,
            "device": device_name,
            "zone": zone_name,
            "crop_relative_paths_by_timestamp": {} # This will be populated
        }
        
        # For summary_track_details_for_db
        current_track_summary_info = {
            "track_metadata_start_time_utc": track_doc.start_time, # Store as datetime for DB
            "track_metadata_end_time_utc": track_doc.end_time,   # Store as datetime for DB
            "device": device_name,
            "zone": zone_name,
            "crop_count": 0 # Will be incremented
        }

        metadata_for_this_track = trackid_to_all_metadocs.get(track_id_str, [])
        metadata_for_this_track.sort(key=lambda m: m.time_stamp)
        
        crop_count_for_this_track = 0

        for meta_doc in metadata_for_this_track:
            track_specific_info = meta_doc.track_ids_info.get(track_id_str)
            if not track_specific_info: continue
            
            bbox = track_specific_info.get("bbox")
            if not bbox or len(bbox) != 4: continue

            video_path_source = meta_doc.evidence_path or meta_doc.video_path
            if not video_path_source: continue
            
            frame_no_str = meta_doc.evidence_frame_number or str(meta_doc.frame_number)
            try: frame_no = int(frame_no_str)
            except (ValueError, TypeError): 
                logger.warning(f"Invalid frame number '{frame_no_str}' for Meta {meta_doc.id if hasattr(meta_doc,'id') else 'N/A'}. Skipping.")
                continue

            ts_obj_utc = meta_doc.time_stamp.astimezone(timezone.utc) if meta_doc.time_stamp.tzinfo is None else meta_doc.time_stamp
            ts_str_filename = ts_obj_utc.strftime("%Y%m%d%H%M%S%f")[:-3]
            out_dir_for_track = base_dir / device_name / zone_name / track_id_str
            crop_filename = f"{ts_str_filename}_{track_id_str}.jpg"
            output_crop_path = out_dir_for_track / crop_filename
            
            job = { "frame_number": frame_no, "bbox": bbox, "output_path": str(output_crop_path) }
            
            crops_by_video_path_key[video_path_source].append(job)
            total_crops_planned += 1
            crop_count_for_this_track += 1
            
            relative_path = str(output_crop_path.relative_to(base_dir))
            current_track_full_info["crop_relative_paths_by_timestamp"][ts_obj_utc.isoformat()] = relative_path
        
        current_track_summary_info["crop_count"] = crop_count_for_this_track
        
        if crop_count_for_this_track > 0 or not metadata_for_this_track : # Store if crops or if it was a track we looked for
            full_manifest_data[track_id_str] = current_track_full_info
            summary_track_details_for_db[track_id_str] = current_track_summary_info
        elif not metadata_for_this_track and track_id_str not in summary_track_details_for_db : # Ensure all initial tracks are in summary even if no metadata
            summary_track_details_for_db[track_id_str] = current_track_summary_info


    logger.info(f"Prepared {total_crops_planned} crop jobs from {len(crops_by_video_path_key)} unique videos.")
    return crops_by_video_path_key, full_manifest_data, summary_track_details_for_db, total_crops_planned, None


def process_crops_and_archive(crops_by_video: dict, base_dir: Path, conn_str: str, cont_name: str):
    if not crops_by_video:
        logger.info("No crop jobs to process.")
        return None, 0, "No crop jobs"

    num_workers = min(cpu_count(), 4)
    logger.info(f"Starting multiprocessing for cropping with {num_workers} workers...")
    total_crops_generated_count = 0
    tasks_for_pool = list(crops_by_video.items())

    try:
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(extract_and_save_crops_from_video, tasks_for_pool),
                                total=len(tasks_for_pool),
                                desc="Processing Videos for Crops", disable=not tasks_for_pool))
            total_crops_generated_count = sum(filter(None, results)) 
    except Exception as e:
        logger.error(f"Error during multiprocessing for crop extraction: {e}")
        return None, total_crops_generated_count, str(e) 

    logger.info(f"Cropping finished. Total crops generated/verified: {total_crops_generated_count}. Results saved in: {base_dir.resolve()}")

    # return 0

    archive_path = create_tar_gz_archive(base_dir)
    if not archive_path:
        logger.warning("Archive creation failed or folder was empty. No upload to Azure.")
        return None, total_crops_generated_count, "Archive creation failed or empty"

    blob_url = upload_to_azure_blob(archive_path, conn_str, cont_name)
    if not blob_url:
        return None, total_crops_generated_count, "Azure upload failed"
        
    return blob_url, total_crops_generated_count, None

def connect_to_mongodb(db_uri, db_alias="default"):
    try: me.disconnect(alias=db_alias)
    except me.errors.NotRegistered: pass
    try:
        me.connect(host=db_uri, alias=db_alias)
        logger.info(f"Successfully connected to MongoDB.")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

def run_enhanced_crop_pipeline(end_time_ist_str: str | None = None, duration_minutes: int = 30):
    run_start_time = datetime.now(timezone.utc)
    log_entry = TrackIdCropsRun(
        run_start_timestamp=run_start_time,
        query_end_time_ist_str=end_time_ist_str if end_time_ist_str else "CurrentTime",
        query_duration_minutes=duration_minutes,
        status="started"
    )
    
    final_status = "failed" 
    error_msg_pipeline = None
    blob_url_final = None

    try:
        connect_to_mongodb(MONGODB_URI)
        try:
            log_entry.save()
            logger.info(f"Initial log entry saved with ID: {log_entry.id}")
        except Exception as e:
            logger.error(f"CRITICAL: Could not make initial save of log_entry: {e}")

        track_id_docs, query_start_utc, query_end_utc, err = get_track_ids_in_window(end_time_ist_str, duration_minutes)
        if query_start_utc: log_entry.query_window_start_utc = query_start_utc
        if query_end_utc: log_entry.query_window_end_utc = query_end_utc

        if err: error_msg_pipeline = f"get_track_ids: {err}"; raise Exception(error_msg_pipeline)
        
        log_entry.total_track_ids_found = len(track_id_docs)
        if not track_id_docs:
            final_status = "no_track_ids_found"; raise Exception(final_status)

        trackid_to_all_metadocs, err = get_metadata_for_tracks(track_id_docs) # Name changed for clarity
        if err: error_msg_pipeline = f"get_metadata: {err}"; raise Exception(error_msg_pipeline)
        
        current_total_metadata_entries = sum(len(docs) for docs in trackid_to_all_metadocs.values())
        log_entry.total_metadata_entries_processed = current_total_metadata_entries

        if not current_total_metadata_entries and log_entry.total_track_ids_found > 0:
             final_status = "no_metadata_found_for_tracks_after_processing"; raise Exception(final_status)
        elif not current_total_metadata_entries :
             final_status = "no_metadata_found"; raise Exception(final_status)

        run_identifier_ts = (query_end_utc or run_start_time).strftime('%Y%m%d_%H%M%S')
        base_dir_for_crops = Path("Track_id_results_optimized") / f"run_{run_identifier_ts}" # Ensure this "results" dir exists or is fine to be created by script
        base_dir_for_crops.mkdir(parents=True, exist_ok=True)

        # prepare_crop_jobs_and_info now returns full_manifest_data and summary_track_details_for_db
        crops_by_video, full_manifest_data, summary_track_details_for_db, total_planned, err = prepare_crop_jobs_and_info(
            track_id_docs, trackid_to_all_metadocs, base_dir_for_crops
        )
        if err: error_msg_pipeline = f"prepare_crop_jobs: {err}"; raise Exception(error_msg_pipeline)
        
        # Store the SUMMARY in the MongoDB log entry
        log_entry.processed_track_ids_summary = summary_track_details_for_db
        
        if not crops_by_video or total_planned == 0:
            log_entry.total_crops_generated = 0 
            final_status = "no_valid_crops_to_process_from_metadata"
            # Still save manifest even if no crops, it might contain track info
            manifest_file_path = base_dir_for_crops / "manifest.json"
            try:
                with open(manifest_file_path, 'w') as f:
                    json.dump(full_manifest_data, f, indent=4)
                logger.info(f"Empty run manifest saved to {manifest_file_path}")
            except Exception as json_e:
                logger.error(f"Failed to save empty run manifest.json: {json_e}")
            raise Exception(final_status)

        # Save the FULL manifest to a JSON file in the base directory (to be archived)
        manifest_file_path = base_dir_for_crops / "manifest.json"
        try:
            with open(manifest_file_path, 'w') as f:
                json.dump(full_manifest_data, f, indent=4)
            logger.info(f"Full manifest data saved to {manifest_file_path}")
        except Exception as json_e:
            logger.error(f"Failed to save manifest.json: {json_e}")
            # Decide if this is a critical error to halt the pipeline
            error_msg_pipeline = (error_msg_pipeline + "; " if error_msg_pipeline else "") + f"manifest_save_error: {json_e}"
            # Not raising exception here, allow archiving of crops to proceed if possible

        blob_url_final, crops_made, err_process = process_crops_and_archive(
            crops_by_video, base_dir_for_crops, connection_string, container_name
        ) # create_tar_gz_archive will now pick up manifest.json
        log_entry.total_crops_generated = crops_made
        if err_process:
            error_msg_pipeline = (error_msg_pipeline + "; " if error_msg_pipeline else "") + f"process_crops_and_archive: {err_process}"

        log_entry.archive_blob_url = blob_url_final
        if blob_url_final:
            final_status = "completed"
            logger.info(f"Pipeline completed successfully. Archive URL: {blob_url_final}")
        elif crops_made > 0: 
             final_status = "completed_partial_archive_upload_failed"
             logger.warning(f"Crops generated, but archive/upload failed: {error_msg_pipeline or 'Unknown archive/upload error'}")
        else: 
            final_status = "failed_during_cropping_or_archiving"
            logger.error(f"Cropping or archiving failed: {error_msg_pipeline or 'Unknown processing error'}")
    except Exception as e:
        if not error_msg_pipeline:
            error_msg_pipeline = str(e)
            logger.error(f"Pipeline processing error: {e}", exc_info=False)
        else:
            logger.error(f"Pipeline halting due to earlier error: {error_msg_pipeline}")
        if final_status not in ["no_track_ids_found", "no_metadata_found_for_tracks_after_processing", 
                                "no_metadata_found", "no_valid_crops_to_process_from_metadata", 
                                "failed_during_cropping_or_archiving", "completed_partial_archive_upload_failed"]:
             final_status = "failed" 
    finally:
        log_entry.status = final_status
        log_entry.error_message = error_msg_pipeline
        log_entry.run_end_timestamp = datetime.now(timezone.utc)
        if log_entry.run_start_timestamp : 
             log_entry.processing_duration_seconds = (log_entry.run_end_timestamp - log_entry.run_start_timestamp).total_seconds()
        else:
            log_entry.processing_duration_seconds = -1 
        try:
            log_entry.save() 
            logger.info(f"Final log entry saved/updated for ID: {log_entry.id}, Status: {final_status}")
        except Exception as db_e: # Catch specific error if save fails again
            logger.error(f"CRITICAL: Failed to save final log entry to DB for ID {log_entry.id if hasattr(log_entry, 'id') and log_entry.id else 'Unknown'}: {db_e}")
            if "document too large" in str(db_e).lower():
                 logger.error("CRITICAL: The TrackIdCropsRun document is STILL too large even after modifications. Further reduction of stored data is needed.")


    return blob_url_final

if __name__ == "__main__":
    logger.info("Starting ENHANCED & OPTIMIZED crop pipeline script...")
    # Example based on your last log:
    # result_url = run_enhanced_crop_pipeline(end_time_ist_str="2025-05-27T17:40:00", duration_minutes=10)
    # This was from a previous log (UTC was 12:00 to 12:10), converting IST 17:40 (UTC 12:10)
    # Let's use a generic example or the one from your latest output if it's different.
    # The latest output implies a run for "2025-05-27 12:00:00+00:00 and 2025-05-27 12:10:00+00:00" (UTC)
    # This corresponds to end_time_ist_str="2025-05-27T17:40:00" (IST for 12:10 UTC) with duration 10 minutes.
    result_url = run_enhanced_crop_pipeline(end_time_ist_str="2025-05-27T19:10:00", duration_minutes=60)


    if result_url:
        print(f"\n--- Main script finished ---\nResult archive is available at: {result_url}")
    else:
        print("\n--- Main script finished ---\nPipeline completed. No archive URL was returned (check logs for status and details).")