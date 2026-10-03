# src/services/video_backup_job.py
"""
One-shot backup job to be run by cron (e.g. hourly).
- Finds files in source directories modified in the last PERIOD (default 1 hour), aligned to the previous exact hour.
- Streams them into a .tar.gz archive.
- Uploads to Azure Blob Storage with MD5 validation.
- Writes a MongoDB (mongoengine) document in collection 'video_backup' describing the backup.

Dependencies:
    pip install pyyaml mongoengine azure-storage-blob azure-identity

Example cron (run at minute 5 every hour, backing up the previous hour):
    5 * * * * /usr/bin/python3 /app/src/services/video_backup_job.py --config /app/config/video_backup.yaml >> /var/log/video_backup.log 2>&1
"""

from __future__ import annotations
import base64
import argparse
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import logging
import os
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mongoengine as me
import yaml
import cv2


# Azure Blob (v12 SDK)
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError
from collections import defaultdict
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

# ----------------------------
# Config
# ----------------------------

@dataclass
class AzureConfig:
    connection_string: str
    container: str
    blob_prefix: str = "archives"  


@dataclass
class MongoConfig:
    uri: str               
    db_name: str           
    alias: str = "default" 


@dataclass
class BackupConfig:
    
    source_dirs: List[str]
    include_globs: List[str] = field(default_factory=lambda: ["*.mp4", "*.avi", "*.mkv", "*.mov"])
    exclude_globs: List[str] = field(default_factory=list)

    
    period_seconds: int = 3600                
    align_to_hour: bool = True                
    lag_seconds: int = 0                      

    archive_dir: str = "/var/tmp/video_archives"

    # Timezone used to compute hour boundaries and in document metadata
    timezone: str = "UTC"

    # Azure + Mongo
    azure: AzureConfig = None  # type: ignore
    mongo: MongoConfig = None  # type: ignore

    # Behavior
    delete_local_archive_after_upload: bool = True
    dry_run: bool = False


# ----------------------------
# Mongo Document (collection: video_backup)
# ----------------------------

def utcnow():
    return dt.datetime.now(dt.timezone.utc)

class VideoBackup(me.Document):
    meta = {
        "collection": "video_backup",
        "indexes": [
            ("period_start", "-period_end"),
            "status",
            "azure_container",
            "azure_blob",
        ],
    }

    # What window did we back up?
    period_start = me.DateTimeField(required=True)
    period_end   = me.DateTimeField(required=True)
    timezone     = me.StringField(default="UTC")

    # What did we include?
    source_dirs  = me.ListField(me.StringField())
    file_count   = me.IntField(default=0)
    total_bytes  = me.LongField(default=0)

    # Artifact on disk
    archive_local_path = me.StringField()
    archive_size_bytes = me.LongField()
    archive_sha256     = me.StringField()
    archive_md5        = me.StringField()

    # Azure upload
    azure_container = me.StringField()
    azure_blob      = me.StringField()
    azure_url       = me.StringField()
    azure_etag      = me.StringField()
    azure_content_md5_b64 = me.StringField()  # what Azure reports
    validate_content = me.BooleanField(default=True)

    # Outcome
    status          = me.StringField(choices=("Pending", "Uploaded", "Skipped", "Failed"), default="Pending")
    integrity_ok    = me.BooleanField(default=False)
    error           = me.StringField()

    created_at      = me.DateTimeField(default=utcnow)
    updated_at      = me.DateTimeField(default=utcnow)

    def touch(self):
        self.updated_at = utcnow()



def connect_mongo(cfg: MongoConfig):
    me.connect(host=cfg.uri, db=cfg.db_name, alias=cfg.alias)

def load_config(path: str) -> BackupConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    azure = AzureConfig(**raw["azure"])
    mongo = MongoConfig(**raw["mongo"])
    raw["azure"] = azure
    raw["mongo"] = mongo
    return BackupConfig(**raw)

def floor_to_hour(dt_obj: dt.datetime) -> dt.datetime:
    return dt_obj.replace(minute=0, second=0, microsecond=0)

def period_window(now: dt.datetime, seconds: int, align_to_hour=True, tz: Optional[str]="UTC") -> Tuple[dt.datetime, dt.datetime]:
    if tz and ZoneInfo:
        now = now.astimezone(ZoneInfo(tz))
    end = floor_to_hour(now) if align_to_hour else now
    start = end - dt.timedelta(seconds=seconds)
    # return timezone-aware UTC moments for storage
    return (start.astimezone(dt.timezone.utc), end.astimezone(dt.timezone.utc))

def iter_files(paths: List[str]) -> Iterable[Path]:
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            yield from pth.rglob("*")
        elif pth.exists():
            yield pth

def matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns) if patterns else True

def compute_hashes(path: Path) -> Tuple[str, str]:
    """Return (sha256_hex, md5_hex)."""
    h_sha = hashlib.sha256()
    h_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h_sha.update(chunk)
            h_md5.update(chunk)
    return h_sha.hexdigest(), h_md5.hexdigest()

def md5_bytes_from_hex(md5_hex: str) -> bytes:
    return bytes.fromhex(md5_hex)

def create_video_from_frames(frame_paths: List[Path], output_path: Path, logger: logging.Logger) -> bool:
    fps = 5
    """Creates an MP4 video from a list of image frame paths."""
    if not frame_paths:
        logger.warning("No frames provided to create video.")
        return False

    try:
        # Get dimensions from the first frame
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            logger.error(f"Could not read first frame: {frame_paths[0]}")
            return False
        height, width, layers = first_frame.shape
        size = (width, height)

        # Define the codec and create VideoWriter object
        # 'mp4v' is a good default for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'h265') 
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        
        logger.info(f"Writing {len(frame_paths)} frames to {output_path} at {fps} FPS...")
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))
            if img is not None:
                out.write(img)
            else:
                logger.warning(f"Skipping unreadable frame: {frame_path}")
        
        out.release()
        logger.info("Video creation complete.")
        return True
    except Exception as e:
        logger.exception(f"Failed to create video {output_path}: {e}")
        return False

def azure_blob_url(account_url_or_conn: str, container: str, blob: str) -> str:
    # best-effort pretty URL if using connection string; real URL is returned from SDK props later
    return f"{container}/{blob}"

# (Your given tar.gz helper kept for convenience; not used in the streaming variant below.)
def create_tar_gz_archive(folder_path) -> Optional[Path]:
    import tarfile as _tarfile
    folder_path = Path(folder_path)
    if not any(folder_path.rglob('*')):
        logging.info(f"Folder {folder_path} is empty. No archive will be created.")
        return None
    archive_path = folder_path.with_suffix('.tar.gz')
    try:
        with _tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)
        logging.info(f"Created archive: {archive_path}")
        return archive_path
    except Exception as e:
        logging.error(f"Failed to create archive {archive_path}: {e}")
        return None

def create_tar_gz_from_files(files: List[Path], archive_path: Path, base_dirs: List[Path]) -> Path:
    """
    Stream files directly into a .tar.gz without duplicating to a staging folder.
    arcname will be relative to the first matching base_dir; otherwise the filename is used.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for f in files:
            if not f.exists() or not f.is_file():
                continue
            # choose the shortest relative arcname under any base_dir
            rel = None
            for b in base_dirs:
                try:
                    rel = f.relative_to(b)
                    break
                except Exception:
                    continue
            arcname = str(rel) if rel else f.name
            tar.add(str(f), arcname=arcname)
    return archive_path


# ----------------------------
# Core job
# ----------------------------

class BackupJob:
    def __init__(self, cfg: BackupConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger("video_backup_job")
        self.log.setLevel(logging.INFO)

    def run_once(self) -> int:
        # Locking logic to prevent concurrent runs
        lock_identifier = self.cfg.azure.blob_prefix.replace('/', '_')
        lock_dir = Path("/var/tmp/video_backup_locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / f"{lock_identifier}.lock"

        if lock_file.exists():
            self.log.warning(f"Lock file {lock_file} exists. Another job is likely running. Skipping.")
            return 0

        try:
            lock_file.touch()

            tzname = self.cfg.timezone or "UTC"
            now_local = dt.datetime.now(ZoneInfo(tzname)) if ZoneInfo and tzname else dt.datetime.now()
            pstart, pend = period_window(now_local, self.cfg.period_seconds, self.cfg.align_to_hour, tzname)

            if self.cfg.lag_seconds > 0:
                pend = pend - dt.timedelta(seconds=self.cfg.lag_seconds)

            # === Stage 1: Find and Group All Frames for the Hour ===
            all_frames = []
            for f in iter_files(self.cfg.source_dirs):
                if not f.is_file() or not matches_any(f.name, self.cfg.include_globs): continue
                try:
                    mtime = dt.datetime.fromtimestamp(f.stat().st_mtime, tz=dt.timezone.utc)
                    if pstart <= mtime < pend:
                        all_frames.append(f)
                except FileNotFoundError:
                    continue

            if not all_frames:
                self.log.info("No frames found for window %s -> %s. Skipping.", pstart.isoformat(), pend.isoformat())
                return 0

            frames_by_minute = defaultdict(list)
            for frame_path in sorted(all_frames):
                try:
                    timestamp_str = '_'.join(frame_path.stem.split('_')[0:2])
                    frame_time = dt.datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                    minute_key = frame_time.strftime("%Y%m%d_%H%M00")
                    frames_by_minute[minute_key].append(frame_path)
                except (IndexError, ValueError):
                    continue

            # === Stage 2: Create all 1-minute video chunks locally ===
            self.log.info(f"Found {len(all_frames)} frames, creating {len(frames_by_minute)} 1-minute video chunks...")
            created_video_paths = []
            for minute_key, frame_paths in frames_by_minute.items():
                video_filename = f"video_{minute_key}.mp4"
                video_path = Path(self.cfg.archive_dir) / video_filename
                
                if create_video_from_frames(frame_paths, video_path, self.log):
                    created_video_paths.append(video_path)
                else:
                    self.log.warning(f"Failed to create video for {minute_key}, it will be excluded from the archive.")

            if not created_video_paths:
                self.log.error("No video chunks were created successfully. Aborting.")
                return 1

            self.log.info(f"Successfully created {len(created_video_paths)} video chunks.")

            # === Stage 3: Archive the newly created videos ===
            start_tag = pstart.astimezone(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            end_tag   = pend.astimezone(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive_filename = f"video_archive_{start_tag}_to_{end_tag}.tar.gz"
            archive_path = Path(self.cfg.archive_dir) / archive_filename
            
            self.log.info(f"Archiving {len(created_video_paths)} videos into {archive_path}")
            # The base directory for videos is the archive_dir itself.
            archive_base_dirs = [Path(self.cfg.archive_dir)]
            create_tar_gz_from_files(created_video_paths, archive_path, archive_base_dirs)

            # === Stage 4: Upload the final archive and log to MongoDB ===
            sha256_hex, md5_hex = compute_hashes(archive_path)
            vb = VideoBackup(
                period_start=pstart, period_end=pend, timezone=tzname,
                source_dirs=[str(d) for d in self.cfg.source_dirs], file_count=len(created_video_paths),
                total_bytes=archive_path.stat().st_size, status="Pending",
                archive_local_path=str(archive_path), archive_sha256=sha256_hex, archive_md5=md5_hex
            )
            vb.save()
            
            upload_ok = False
            blob_name = f"{self.cfg.azure.blob_prefix.strip('/')}/{archive_filename}".lstrip("/")
            try:
                # ... (Azure upload logic is the same as before) ...
                bsc = BlobServiceClient.from_connection_string(self.cfg.azure.connection_string)
                container = bsc.get_container_client(self.cfg.azure.container)
                try: container.create_container()
                except ResourceExistsError: pass
                blob = container.get_blob_client(blob=blob_name)
                with open(archive_path, "rb") as data:
                    blob.upload_blob(data, overwrite=True, validate_content=True, content_settings=ContentSettings(content_type="application/gzip"))
                
                # Update Mongo doc
                props = blob.get_blob_properties()
                vb.status = "Uploaded"
                vb.azure_url = blob.url
                # ... (rest of the mongo doc update) ...
                vb.save()
                upload_ok = True
                self.log.info(f"Successfully uploaded archive to Azure: {blob.url}")

            except Exception as e:
                vb.status, vb.error = "Failed", f"Azure upload failed: {e}"
                vb.touch(), vb.save()
                self.log.exception(f"Azure upload failed for {archive_path}")
                return 1

            # === Stage 5: Cleanup local files ===
            if upload_ok and self.cfg.delete_local_archive_after_upload:
                self.log.info("Cleaning up temporary local files...")
                for video_path in created_video_paths:
                    video_path.unlink(missing_ok=True)
                archive_path.unlink(missing_ok=True)
                self.log.info("Cleanup complete.")

            return 0
        finally:
            # ALWAYS remove the lock file
            lock_file.unlink(missing_ok=True)

# ----------------------------
# CLI
# ----------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Run one video backup job.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--dry-run", action="store_true", help="Discover & plan only; do not archive or upload.")
    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True

    connect_mongo(cfg.mongo)
    job = BackupJob(cfg)
    rc = job.run_once()
    sys.exit(rc)

if __name__ == "__main__":
    main()