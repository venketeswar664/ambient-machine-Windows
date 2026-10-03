import mongoengine as me
from datetime import datetime, timedelta
from pathlib import Path
import os
import cv2
import sys
import tarfile
import traceback
import datetime as dt
from multiprocessing import Pool, cpu_count
import mongoengine as db
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField
from collections import defaultdict
from azure.storage.blob import BlobServiceClient
from pathlib import Path
import logging
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# === Your Schemas (same as you provided) ===
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
    createdAt = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))
    updatedAt = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))


class Zone(db.Document):
    zone_id = db.StringField(required=True, unique=True)
    name = db.StringField(required=True)
    zone_type = db.StringField(required=True)
    store = db.ReferenceField('Stores')
    colourHex = db.StringField(default="#09467c")
    camera_zone_mappings = db.ListField(db.ReferenceField('CameraZoneMapping'))
    createdAt = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))
    updatedAt = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))


class Services(db.Document):
    service_name = db.StringField(required=True, unique=True)
    descriptions= db.StringField()
    zones = db.ListField(db.ReferenceField('Zone'))
    pipeline  = db.ReferenceField('Pipeline')
    fixed_zone = db.BooleanField()


class Cameras(db.Document):
    device_name = db.StringField(required=True, unique=True)
    store = db.ReferenceField("Stores")
    camera_address = db.StringField(required=True)
    pipelines = db.ReferenceField("Pipeline")
    services = db.ReferenceField("Services")
    zones = db.ListField(db.ReferenceField('Zone'))
    process_skip_frame = db.IntField(default=5)
    active = db.BooleanField(default=True)
    save_plotted_video = db.BooleanField(default=False)
    save_raw_video = db.BooleanField(default=False)
    status = db.StringField()
    created_at = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))
    updated_at = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))
    meta = {'collection': 'cameras_test'}
    def __str__(self):
        return f"Camera {self.device_name}"


class CameraZoneMapping(db.Document):
    camera = db.ReferenceField('Cameras')
    zone = db.ReferenceField('Zone')
    roi = db.ListField(db.ListField(db.FloatField()))
    createdAt = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))
    updatedAt = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))


class Pipeline(db.Document):
    name = db.StringField(required=True, unique=True)
    path = db.StringField(required=True)
    created_at = db.DateTimeField(default=dt.datetime.now(dt.timezone.utc))


class Metadata(Document):
    frame_number = IntField(required=True)
    time_stamp = DateTimeField(required=True)
    video_path = StringField(required=True)
    evidence_path = StringField(required=False)
    device_name = StringField(required=True)
    track_ids_info = DictField(
        field=DictField(
            fields={
                "track_id": StringField(required=True),
                "bbox": ListField(IntField(), required=True),
                "confidence": FloatField(required=True),
                "label": IntField(required=True),
                "label_name": StringField(required=True),
                "instance_dict": DictField()
            }
        )
    )
    meta = {'collection': 'metadata', 'indexes': ['time_stamp']}


class Entry_exit(Document):
    frame_number = IntField(required=True)
    time_stamp = DateTimeField(required=True)
    video_path = StringField(required=True)
    evidence_path = StringField(required=False)
    device_name = StringField(required=True)
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
    meta = {'collection': 'entry_exit', 'indexes': ['time_stamp']}


class Security(Document):
    camera = ReferenceField('Cameras')
    time_stamp = DateTimeField(default=dt.datetime.now(dt.timezone.utc))
    lights = BooleanField(required=True)
    camera_tampering = BooleanField(required=True)
    smoke = BooleanField(required=True)
    fire = BooleanField(required=True)


class TrackIDMetadata(me.Document):
    meta = {'collection': 'track_id_metadata'}

    track_id = LongField(required=True, regex=r"^\s*\S.*\S\s*$")  # trim effect with regex
    device = StringField(required=True)
    attended = IntField(required=True)
    dwell_time = LongField(required=True)
    start_time = DateTimeField(required=True)
    end_time = DateTimeField(required=True)
    gender = StringField()
    label = StringField()
    zone = StringField(required=True)
    interacted = IntField(required=True)
    interacted_list = ListField(StringField(), default=[])
    
def upload_to_azure_blob(file_path: Path, connection_string: str, container_name: str) -> str | None:
    """
    Uploads a file to Azure Blob Storage and returns the blob URL if successful, else None.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Ensure container exists (creates if not)
        try:
            container_client.create_container()
        except Exception:
            # Container likely already exists, ignore error
            pass

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

    

def extract_and_save_crops_from_video(args):
    """
    Multiprocessing worker function to extract and save crops from a single video.
    args: tuple (video_path, crops_list)
    """
    video_path, crops_list = args
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return

    pbar = tqdm(crops_list, desc=f"Cropping {Path(video_path).name}", unit="frame", leave=False)
    for job in pbar:
        frame_no = job['frame_number']
        bbox = job['bbox']
        out_path = job['output_path']

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {frame_no} from {video_path}")
            continue

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1p = max(0, x1 - 10)
        y1p = max(0, y1 - 10)
        x2p = min(w, x2 + 10)
        y2p = min(h, y2 + 10)

        if x1p >= x2p or y1p >= y2p:
            logger.warning(f"Invalid bbox for frame {frame_no} in {video_path}")
            continue

        crop_img = frame[y1p:y2p, x1p:x2p]

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop_img)
        except Exception as e:
            logger.error(f"Failed to save crop {out_path}: {e}")

    cap.release()
    
def create_tar_gz_archive(folder_path: Path):
    archive_path = folder_path.with_suffix('.tar.gz')
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)
        logger.info(f"Created archive: {archive_path}")
    except Exception as e:
        logger.error(f"Failed to create archive {archive_path}: {e}")



def run_crop_pipeline(hours_ago=1):
    try:
        me.disconnect()
        me.connect(db="sentinel", host="mongodb://sentinel:sentinelMongo_test123@localhost:27017/sentinel", alias="default")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        sys.exit(1)

    now = dt.datetime.now(dt.timezone.utc)
    window_start = now - timedelta(hours=hours_ago)

    logger.info(f"Processing TrackIDMetadata from {window_start} to {now}")

    # Query TrackIDMetadata documents
    collection = TrackIDMetadata._get_collection()
    raw_tracks = list(collection.find({"start_time": {"$gte": window_start, "$lt": now}}))
    if not raw_tracks:
        logger.info("No TrackIDMetadata found in the time window.")
        return

    all_track_ids = [str(t['track_id']) for t in raw_tracks]
    meta_time_start = min(t['start_time'] for t in raw_tracks)
    meta_time_end = max(t['end_time'] for t in raw_tracks)

    # Fetch all Metadata documents in the broad time range
    metas = Metadata.objects(time_stamp__gte=meta_time_start, time_stamp__lte=meta_time_end)

    # Map track_id to Metadata docs for quick lookup
    trackid_to_metadocs = {tid: [] for tid in all_track_ids}
    for meta in metas:
        keys = meta.track_ids_info.keys()
        for tid in all_track_ids:
            if tid in keys:
                trackid_to_metadocs[tid].append(meta)
                
    # for each track, only keep the first metadata per unique second ───
    for tid, docs in trackid_to_metadocs.items():
        # sort by increasing timestamp so you pick the earliest 5 each second
        docs.sort(key=lambda m: m.time_stamp)
        filtered = []
        counts_per_sec = defaultdict(int)

        for m in docs:
            # floor to the second
            sec = m.time_stamp.replace(microsecond=0)
            if counts_per_sec[sec] < 5:
                counts_per_sec[sec] += 1
                filtered.append(m)
            # once counts_per_sec[sec] hits 5, further docs in that second are skipped

        trackid_to_metadocs[tid] = filtered
        
    base_dir = Path("results") / f"run_{now.strftime('%Y%m%d_%H%M%S')}"
    base_dir.mkdir(parents=True, exist_ok=True)

    crops_by_video = {}

    # Progress bar for tracks
    for track_doc in tqdm(raw_tracks, desc="Processing Tracks", unit="track"):
        track_id_str = str(track_doc['track_id'])
        zone_name    = track_doc.get('zone', 'unknown')
        device_name  = track_doc.get('device', 'unknown')

        logger.info(f"Processing TrackID {track_id_str}, Zone '{zone_name}', Device '{device_name}'")

        for meta in trackid_to_metadocs.get(track_id_str, []):
            info = meta.track_ids_info.get(track_id_str)
            if not info:
                continue
            bbox = info.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            video_path = meta.evidence_path or meta.video_path
            if not video_path:
                logger.warning(f"No video path for metadata {meta.id}, skipping")
                continue

            frame_no = meta.frame_number
            # e.g. "20250524123456789" (YYYYMMDDhhmmssfff)
            ts_str = meta.time_stamp.strftime("%Y%m%d%H%M%S%f")[:-3]
            print(track_id_str[11:])
            # —––– HERE’S THE CHANGE –––––—
            out_dir = base_dir / device_name / zone_name / track_id_str[11:]

            # new filename: timestamp_trackid.jpg
            filename = f"{ts_str}_{track_id_str}.jpg"

            out_path = out_dir / filename

            job = {
                "frame_number": int(frame_no),
                "bbox": bbox,
                "output_path": out_path
            }
            crops_by_video.setdefault(video_path, []).append(job)


    # Use multiprocessing to extract crops from videos in parallel
    num_workers = min(cpu_count(), 4)
    logger.info(f"Starting multiprocessing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(extract_and_save_crops_from_video, crops_by_video.items()),
                  total=len(crops_by_video),
                  desc="Cropping videos"))

    logger.info(f"Cropping finished. Results saved in: {base_dir.resolve()}")
    create_tar_gz_archive(base_dir)
    archive_path = base_dir.with_suffix('.tar.gz')
    blob_url = upload_to_azure_blob(archive_path, connection_string, container_name)
    if blob_url:
        logger.info(f"Archive uploaded successfully. Blob URL: {blob_url}")
    else:
        logger.error("Failed to upload archive to Azure Blob Storage.")
        
    return blob_url


if __name__ == "__main__":
    url = run_crop_pipeline()
    if url:
        print(f"Result archive is available at: {url}")
    else:
        print("Pipeline completed but upload failed or did not run.")                                                                                                                                               
