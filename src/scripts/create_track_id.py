import argparse
import datetime
from collections import Counter, defaultdict
import mongoengine as me
import sys
from pathlib import Path
import logging
from logging import Logger
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

# Add tqdm import
from tqdm import tqdm
from mongoengine.queryset.visitor import Q


import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField

class Stores(db.Document):
    store_id = db.StringField()
    name = db.StringField()
    organization = db.StringField()
    format = db.StringField()
    category = db.StringField()
    state = db.StringField()
    city = db.StringField()
    district = db.StringField()
    location = db.DictField()
    layout = db.DictField()
    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updatedAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))


class Zone(db.Document):
    zone_id = db.StringField(required=True, unique=True)  # Unique ID for the zone
    name = db.StringField(required=True)  # Zone name (e.g., "entry", "lab", "pathway-1")
    zone_type = db.StringField(required=True)  # Type of zone (e.g., "area", "object")
    store = db.ReferenceField('Stores')  # Store associated with this zone
    colourHex = db.StringField(default="#09467c")  # Zone color for visualization
    camera_zone_mappings = db.ListField(db.ReferenceField('CameraZoneMapping'))  # Multiple cameras can map to this zone
    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updatedAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))

    
class Services(db.Document):
    service_name = db.StringField(required=True, unique=True)
    descriptions= db.StringField()
    zones = db.ListField(db.ReferenceField('Zone'))
    pipeline  = db.ReferenceField('Pipeline')
    fixed_zone = db.BooleanField()

class Cameras(db.Document):
    device_name = db.StringField(required=True, unique=True)  # Format: Camera-001
    store = db.ReferenceField("Stores")  # Store associated with this camera
    camera_address = db.StringField(required=True)  # Video feed URL or location
    pipelines = db.ReferenceField("Pipeline")  # Processing pipelines
    services = db.ReferenceField("Services")
    zones = db.ListField(db.ReferenceField('Zone'))  # References multiple zone mappings
    process_skip_frame = db.IntField(default=5)  # Skip frame interval for processing
    active = db.BooleanField(default=True)
    save_plotted_video = db.BooleanField(default=False)
    save_raw_video = db.BooleanField(default=False)
    status = db.StringField()
    department = db.StringField()
    frame = db.StringField()

    created_at = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updated_at = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    meta = {
        'collection': 'cameras'
    }
    def __str__(self):
        return f"Camera {self.device_name}"
    
class CameraZoneMapping(db.Document):
    camera = db.ReferenceField('Cameras')  # The camera capturing this zone
    zone = db.ReferenceField('Zone')  # The actual zone
    roi = db.ListField(db.ListField(db.FloatField()))  # Multiple ROIs per zone in this camera
    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updatedAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))

   
    
class Pipeline(db.Document):
    # Ensure pipeline names are unique if needed
    name = db.StringField(required=True, unique=True)
    path = db.StringField(required=True)
    created_at = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))


class Metadata(Document):
    # Basic Video and Frame Info
    frame_number = IntField(required=True)
    evidence_frame_number = StringField()
    time_stamp = DateTimeField(required=True)
    video_path = StringField(required=False)
    evidence_path = StringField(required=True)
    device_name = StringField(required=True)
    # Track ID Information (nested dictionary with string keys)
    track_ids_info = DictField(
        field=DictField(
            fields={
                "track_id": StringField(required=True),
                "track_id_path_list": StringField(required=True),

                "bbox": ListField(IntField(), required=True),
                "confidence": FloatField(required=True),
                "label": IntField(required=True),
                "label_name": StringField(required=True),
                "instance_dict": DictField()
            }
        )
    )

    meta = {
        'collection': 'metadata',
        'indexes': ['time_stamp']
    }
    
class Entry_exit(Document):
    # Basic Video and Frame Info
    frame_number = IntField(required=True)
    time_stamp = DateTimeField(required=True)
    video_path = StringField(required=True)
    evidence_path = StringField(required=False)
    device_name = StringField(required=True)

    # Track ID Information (nested dictionary with string keys)
    track_ids_info = DictField(
        field=DictField(
            fields={
                "track_id": IntField(required=True),
                "bbox": ListField(IntField(), required=True),
                "confidence": FloatField(required=True),
                "label": IntField(required=True),
                "label_name": StringField(required=True),
                "instance_dict": DictField()
            }
        )
    )

    meta = {
        'collection': 'entry_exit',
        'indexes': ['time_stamp']
    }
    
class Security(Document):
    
    camera = ReferenceField('Cameras')
    time_stamp = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    lights = BooleanField(required=True)
    camera_tampering = BooleanField(required=True)
    smoke = BooleanField(required=True)
    fire = BooleanField(required=True)



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
    interacted = IntField(required=True) # Assuming these fields exist
    interacted_list = ListField(StringField(), default=[]) # Assuming these fields exist
    # Add any other fields from TrackIDMetadata that are actually used by the script implicitly

# MODIFIED TrackIdCropsRun Schema
class TrackIdCropsRun(Document):
    meta = {'collection': 'track_id_crops_runs'}
    run_start_timestamp = DateTimeField(required=True) # Known at start
    
    # Fields modified to allow null initially
    run_end_timestamp = DateTimeField(null=True)
    processing_duration_seconds = FloatField(null=True)
    query_window_start_utc = DateTimeField(null=True)
    query_window_end_utc = DateTimeField(null=True)
    
    query_end_time_ist_str = StringField()
    query_duration_minutes = IntField()
    archive_blob_url = StringField(null=True)
    processed_track_ids_info = DictField(
        field=DictField(
            fields={
                "track_metadata_start_time_utc": DateTimeField(),
                "track_metadata_end_time_utc": DateTimeField(),
                "device": StringField(),
                "zone": StringField(),
                "crop_relative_paths_by_timestamp": DictField(
                    field=StringField()
                )
            }
        ),
        default={} # Provide a default empty dict
    )
    status = StringField(default="pending")
    error_message = StringField(null=True)
    total_track_ids_found = IntField(default=0)
    total_metadata_entries_processed = IntField(default=0)
    total_crops_generated = IntField(default=0)



class TrackIDRecord(Document):
    track_id = StringField(required=True) 
    camera = ReferenceField(Cameras, required=True) 
    videos = ListField(StringField()) # Changed: Now a list of strings
    track_id_path_lists = ListField(ListField(StringField())) # Changed: Now a list of strings

    model_name = StringField(required=True) 
    label = StringField(required=True)
    start_time = DateTimeField() # Will store UTC datetime
    end_time = DateTimeField()   # Will store UTC datetime
    zones = DictField(field=DictField(field=ObjectIdField()))
    created_at = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)) # UTC default
    updated_at = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)) # UTC default

    meta = {
        'collection': 'track_id_records',
        'indexes': [
            {'fields': ('track_id', 'camera', 'model_name'), 'unique': True},
            'camera',
            'model_name',
            'videos', # Index on list of strings
            'start_time',
            'end_time',
        ],
        'auto_create_index': True,
        'ordering': ['-updated_at']
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.datetime.now(datetime.timezone.utc) # Ensure updated_at is UTC
        return super(TrackIDRecord, self).save(*args, **kwargs)

    def __str__(self):
        cam_name = self.camera.name if self.camera and hasattr(self.camera, 'name') else 'N/A'
        return f"TrackIDRecord(track_id='{self.track_id}', camera='{cam_name}', model='{self.model_name}')"

    
# class TrackIDMetadata(Document):
#     meta = {'collection': 'trackid_metadata'}

#     zone = StringField(required=True)              # e.g. "laptop"
#     track_id = LongField(required=True, unique=True)  # Large numeric track ID
#     start_time = DateTimeField(required=True)
#     end_time = DateTimeField(required=True)
#     attended = IntField()
#     device = StringField(required=True)             # e.g. "Camera-005"
#     dwell_time = LongField()
#     label = StringField()
#     interacted = IntField()
#     interacted_list = ListField(StringField())

#     # You can add indexes on track_id and start_time for efficiency
#     meta = {
#         'indexes': [
#             'track_id',
#             'start_time',
#             ('start_time', 'end_time')
#         ]
#     }
    
# Add src path to sys.path
src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))
# Ensure schemas.py is in the same directory or Python path
from src.database.schemas.sentinel_poc_schema import Metadata, TrackIDRecord, Cameras


def setup_logger() -> Logger:
    """
    Configure and return a logger that writes to:
    ./logs/track_ids_creation/<YYYY-MM-DD>/create_track_ids_<YYYYMMDDTHHMMSS>.logs
    """
    # Create directory path based on current date
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y%m%dT%H%M%S")
    log_dir = Path("logs") / "track_ids_creation" / date_str
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"create_track_ids_{timestamp_str}.logs"

    logger = logging.getLogger("track_ids_creation")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console (stream) handler, if you still want to see output on console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fh_formatter)
    logger.addHandler(ch)

    return logger


# --- Helper Functions ---
# Your get_camera_doc, determine_zone_for_track, get_most_frequent_label,
# and calculate_label_for_record functions remain the same, except using logger.


def get_camera_doc(device_name_str, logger: Logger):
    """
    Retrieves or creates a Camera document.
    """
    if not device_name_str:
        logger.warning("Attempted to get/create camera with empty device_name.")
        return None
    camera = Cameras.objects(device_name=device_name_str).first()
    return camera


def determine_zone_for_track(instance_dict_data):
    current_zone_final = "no-zone"
    inside_candidates, on_candidates, other_candidates = [], [], []
    if isinstance(instance_dict_data, dict):
        for zone_name, interaction_data in instance_dict_data.items():
            if not isinstance(interaction_data, dict):
                continue
            iou, location = interaction_data.get("iou", 0.0), interaction_data.get("location")
            if location == "inside":
                inside_candidates.append((iou, zone_name))
            elif location == "on":
                on_candidates.append((iou, zone_name))
            elif iou > 0:
                other_candidates.append((iou, zone_name))
    if inside_candidates:
        current_zone_final = sorted(inside_candidates, key=lambda x: x[0], reverse=True)[0][1]
    elif on_candidates:
        current_zone_final = sorted(on_candidates, key=lambda x: x[0], reverse=True)[0][1]
    elif other_candidates:
        current_zone_final = sorted(other_candidates, key=lambda x: x[0], reverse=True)[0][1]
    return current_zone_final


def get_most_frequent_label(label_names_list):
    if not label_names_list:
        return "unknown"
    counts = Counter(label_names_list)
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else "unknown"


def calculate_label_for_record(track_id_record: TrackIDRecord):
    all_meta_ids = set()
    if isinstance(track_id_record.zones, dict):
        for timestamp_dict in track_id_record.zones.values():
            if isinstance(timestamp_dict, dict):
                all_meta_ids.update(timestamp_dict.values())
    if not all_meta_ids:
        return track_id_record.label if track_id_record.label else "unknown"
    relevant_metadata_docs = Metadata.objects(id__in=list(all_meta_ids)).only("track_ids_info")
    label_names_for_track = []
    for meta_doc in relevant_metadata_docs:
        if isinstance(meta_doc.track_ids_info, dict):
            track_info = meta_doc.track_ids_info.get(track_id_record.track_id)
            if isinstance(track_info, dict):
                label_name = track_info.get("label_name")
                if label_name:
                    label_names_for_track.append(label_name)
    if not label_names_for_track:
        return track_id_record.label if track_id_record.label else "unknown"
    return get_most_frequent_label(label_names_for_track)


# --- Main Processing Logic ---
def process_metadata_for_duration(start_dt_utc, end_dt_utc, model_name_arg, logger: Logger):
    logger.info(f"Processing metadata from {start_dt_utc.isoformat()} to {end_dt_utc.isoformat()} for model '{model_name_arg}'")

    metadata_docs = Metadata.objects(time_stamp__gte=start_dt_utc, time_stamp__lte=end_dt_utc)

    if not metadata_docs:
        logger.info("No metadata found in the specified time range.")
        return
    logger.info(f"Found {len(metadata_docs)} metadata document(s).")

    aggregated_track_data = defaultdict(
        lambda: {
            "timestamps": [],
            "label_names": [],
            "zone_entries": defaultdict(dict),
            "evidence_paths": set(),
            "track_id_path_lists": [],
        }
    )

    for meta_doc in metadata_docs:
        if not isinstance(meta_doc.track_ids_info, dict):
            continue
        meta_time_utc = meta_doc.time_stamp
        if meta_time_utc.tzinfo is None:  # Ensure meta_time from DB is UTC aware
            meta_time_utc = meta_time_utc.replace(tzinfo=datetime.timezone.utc)

        track_id_path_list = None
        for track_id_str, track_data in meta_doc.track_ids_info.items():
            if not isinstance(track_data, dict):
                continue
            label_name, instance_dict = (
                track_data.get("label_name"),
                track_data.get("instance_dict"),
            )
            track_id_path_list = track_data.get("track_id_path_list")


            # print("1 track_id_path_list: ", track_id_path_list)

            if label_name is None or instance_dict is None:
                continue

            aggregation_key = (track_id_str, meta_doc.device_name)
            aggregated_track_data[aggregation_key]["timestamps"].append(meta_time_utc)
            aggregated_track_data[aggregation_key]["label_names"].append(label_name)
            current_zone = determine_zone_for_track(instance_dict)
            timestamp_key = meta_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            aggregated_track_data[aggregation_key]["zone_entries"][current_zone][timestamp_key] = meta_doc.id
            if meta_doc.evidence_path:
                aggregated_track_data[aggregation_key]["evidence_paths"].add(meta_doc.evidence_path)
            if track_id_path_list:
                # print("2 track_id_path_list: ", track_id_path_list)

                aggregated_track_data[aggregation_key]["track_id_path_lists"].append(track_id_path_list)

    logger.info(f"Aggregated data for {len(aggregated_track_data)} unique tracks (track_id, device_name).")

    created_count = 0
    updated_count = 0

    logger.info("Processing aggregated track data to create/update TrackIDRecords...")


    for (track_id_str, device_name), data in tqdm(aggregated_track_data.items(), desc="Processing Tracks"):
        camera_doc = get_camera_doc(device_name, logger)
        if not camera_doc:
            continue

        batch_video_paths = list(data["evidence_paths"])
        track_id_path_lists = list(data["track_id_path_lists"])
        batch_start_time_utc = min(data["timestamps"]) if data["timestamps"] else None
        batch_end_time_utc = max(data["timestamps"]) if data["timestamps"] else None
        batch_zones = {zn: ts_map for zn, ts_map in data["zone_entries"].items()}

        try:
            # Try to get record, else create a new one (don't rely on DoesNotExist)
            record = TrackIDRecord.objects(
                Q(track_id=track_id_str) & Q(camera=camera_doc) & Q(model_name=model_name_arg)
            ).first()

            if not record:
                # Create new
                record = TrackIDRecord(
                    track_id=track_id_str,
                    camera=camera_doc,
                    model_name=model_name_arg,
                    videos=[],
                    track_id_path_lists=[],
                    zones={},
                    label=None
                )
                created_count += 1
            else:
                updated_count += 1

            # --- Start/End time update ---
            def utcify(dt):
                return dt.replace(tzinfo=datetime.timezone.utc) if dt and dt.tzinfo is None else dt

            batch_start_time_utc = utcify(batch_start_time_utc)
            batch_end_time_utc = utcify(batch_end_time_utc)
            if batch_start_time_utc and (not record.start_time or batch_start_time_utc < utcify(record.start_time)):
                record.start_time = batch_start_time_utc
            if batch_end_time_utc and (not record.end_time or batch_end_time_utc > utcify(record.end_time)):
                record.end_time = batch_end_time_utc

            # --- Merge videos ---
            if not isinstance(record.videos, list):
                record.videos = []
            for path_str in batch_video_paths:
                if path_str not in record.videos:
                    record.videos.append(path_str)

            # --- Merge track_id_path_lists ---
            if not isinstance(record.track_id_path_lists, list):
                record.track_id_path_lists = []
            for path_list in track_id_path_lists:
                if path_list not in record.track_id_path_lists:
                    record.track_id_path_lists.append(path_list)

            # --- Merge zones ---
            if not isinstance(record.zones, dict):
                record.zones = {}
            for zone_name, ts_map in batch_zones.items():
                if zone_name not in record.zones or not isinstance(record.zones[zone_name], dict):
                    record.zones[zone_name] = {}
                record.zones[zone_name].update(ts_map)

            # --- Label update ---
            record.label = calculate_label_for_record(record) or get_most_frequent_label(data["label_names"])
            record.save()

        except me.errors.NotUniqueError:
            cam_name_for_error = camera_doc.device_name if camera_doc else "UnknownCamera"
            logger.error(f"NotUniqueError for track_id='{track_id_str}', camera='{cam_name_for_error}', model_name='{model_name_arg}'.")

        except Exception as e:
            logger.error(f"Unexpected error while processing track '{track_id_str}' for device '{device_name}': {e}")
            
    logger.info(f"Processing complete. Created: {created_count}, Updated: {updated_count}.")


if __name__ == "__main__":
    # Initialize logger at the very beginning
    logger = setup_logger()

    load_dotenv()
    default_start_time_str = "2025-06-10T09:30:00"
    default_end_time_str = "2025-06-10T09:50:00"
    default_mongo_uri_from_env = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    parsed_uri = urlparse(default_mongo_uri_from_env)
    default_db_name_from_uri = parsed_uri.path.lstrip("/") if parsed_uri.path else "mydatabase"
    if not default_db_name_from_uri:
        default_db_name_from_uri = "mydatabase"
    default_model_name = ""

    parser = argparse.ArgumentParser(
        description="Process metadata to create/update TrackID records."
    )
    parser.add_argument(
        "--start-time",
        default=default_start_time_str,
        help=f"Start time ISO (Default: {default_start_time_str})",
    )
    parser.add_argument(
        "--end-time",
        default=default_end_time_str,
        help=f"End time ISO (Default: {default_end_time_str})",
    )
    parser.add_argument(
        "--mongo-uri",
        default=default_mongo_uri_from_env,
        help=f"MongoDB URI (Reads from .env MONGODB_URI by default)",
    )
    parser.add_argument(
        "--db-name",
        default=default_db_name_from_uri,
        help=f"MongoDB database name (Default from URI or 'mydatabase')",
    )
    parser.add_argument(
        "--model-name",
        default=default_model_name,
        help=f"Model name (Default: '{default_model_name}')",
    )
    args = parser.parse_args()

    try:
        start_datetime_input = datetime.datetime.fromisoformat(
            args.start_time.replace("Z", "+00:00")
        )
        start_datetime_utc = (
            start_datetime_input.replace(tzinfo=datetime.timezone.utc)
            if start_datetime_input.tzinfo is None
            else start_datetime_input.astimezone(datetime.timezone.utc)
        )
        end_datetime_input = datetime.datetime.fromisoformat(
            args.end_time.replace("Z", "+00:00")
        )
        end_datetime_utc = (
            end_datetime_input.replace(tzinfo=datetime.timezone.utc)
            if end_datetime_input.tzinfo is None
            else end_datetime_input.astimezone(datetime.timezone.utc)
        )
    except ValueError as ve:
        logger.error(f"Invalid datetime format: {ve}. Use YYYY-MM-DDTHH:MM:SS.")
        sys.exit(1)

    mongo_connection_uri = args.mongo_uri
    database_name_to_connect = args.db_name
    logger.info(f"Effective MongoDB URI: {mongo_connection_uri}")
    logger.info(f"Effective Database Name: {database_name_to_connect}")

    try:
        me.connect(
            db=database_name_to_connect,
            host=mongo_connection_uri,
            UuidRepresentation="standard",
        )
        logger.info("Successfully connected to MongoDB.")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        sys.exit(1)

    process_metadata_for_duration(
        start_datetime_utc, end_datetime_utc, args.model_name, logger
    )

    try:
        me.disconnect()
        logger.info("Disconnected from MongoDB.")
    except Exception as e:
        logger.error(f"Error disconnecting: {e}")
    logger.info("Script finished.")
