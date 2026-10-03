# src/services/video_backup_scheduler.py
"""
Generates per-camera YAML configs from a MongoDB database. These configs are
used by the backup_service.py job.

This script is intended to be run by the centralized service_manager.py.

With the --run-now flag, it will also immediately run all generated backup jobs
in parallel threads, which is useful for manual triggers or initial runs.
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional
import datetime as dt
import yaml
import mongoengine as me
from mongoengine.queryset.visitor import Q
from concurrent.futures import ThreadPoolExecutor, as_completed

# No longer need crontab
# try:
#     from crontab import CronTab
# except Exception:
#     CronTab = None

# ---------- defaults / env ----------

ROOT = Path(__file__).resolve().parents[2]  # repo root
# JOB_SCRIPT is no longer needed here, as this script doesn't create cron jobs anymore.
# JOB_SCRIPT = ROOT / "src" / "services" / "backup_service.py"

FRAMES_BASE_DIR = Path(os.environ.get(
    "VIDEO_BACKUP_BASE_DIR",
    "/home/sentinel/Projects/ambient-machine/results/frames",
))

CONFIG_DIR = Path(os.environ.get(
    "VIDEO_BACKUP_CONFIG_DIR",
    ROOT / "src" / "configs" / "backup_service" / "video_backup",
))

LOG_DIR = Path(os.environ.get("VIDEO_BACKUP_LOG_DIR", ROOT / "logs" / "video_backup"))
# DEFAULT_CRON is removed, schedule is managed by service_manager.py
# DEFAULT_CRON = os.environ.get("VIDEO_BACKUP_DEFAULT_CRON", "5 * * * *")
DEFAULT_PERIOD = int(os.environ.get("VIDEO_BACKUP_PERIOD", "3600"))
DEFAULT_LAG = int(os.environ.get("VIDEO_BACKUP_LAG", "0"))
DEFAULT_TZ = os.environ.get("VIDEO_BACKUP_TZ", "Asia/Kolkata")

AZURE_CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER = os.environ.get("AZURE_CONTAINER", "")
AZURE_PREFIX_ROOT = os.environ.get("AZURE_BLOB_PREFIX_ROOT", "")

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://sentinel:sentinelMongo_test123@10.8.0.26:27017/sentinel_warehouse")
MONGO_DB = os.environ.get("MONGODB_DBNAME", "sentinel_warehouse")

# DEFAULT_INCLUDE = ["*.mp4", "*.avi", "*.mkv", "*.mov"]
DEFAULT_INCLUDE = ["*.jpg", "*.jpeg", "*.png"]
DEFAULT_EXCLUDE: List[str] = []

# ---------- Mongo models ----------

class Camera(me.DynamicDocument):
    meta = {"collection": "cameras", "db_alias": "default"}

# ---------- helpers ----------

def _connect_mongo() -> None:
    me.connect(host=MONGO_URI, db=MONGO_DB, alias="default")

def _fetch_cameras() -> List[Camera]:
    return list(Camera.objects(active=True).filter(Q(backup=True) | Q(backup__exists=False)))

def _site_of(cam: Camera) -> str:
    return getattr(cam, "site", None) or "Belapur"

def _device_of(cam: Camera) -> Optional[str]:
    for key in ("deviceName", "device_name", "device", "name"):
        v = getattr(cam, key, None)
        if v:
            return str(v)
    return None

# _cron_of is removed
# def _cron_of(cam: Camera) -> str:
#     return getattr(cam, "backup_cron", None) or DEFAULT_CRON

def _period_of(cam: Camera) -> int:
    return int(getattr(cam, "backup_period_seconds", None) or DEFAULT_PERIOD)

def _timezone_of(cam: Camera) -> str:
    return getattr(cam, "timezone", None) or DEFAULT_TZ

def _include_globs(cam: Camera) -> List[str]:
    return list(getattr(cam, "include_globs", None) or DEFAULT_INCLUDE)

def _exclude_globs(cam: Camera) -> List[str]:
    return list(getattr(cam, "exclude_globs", None) or DEFAULT_EXCLUDE)

def _source_dir_for(cam: Camera, device: str) -> Path:
    """Finds the latest .../YYYY-MM-DD/<device>/raw directory."""
    # Find the most recent date-stamped folder
    dated_dirs = sorted(
        [p for p in FRAMES_BASE_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]") if p.is_dir()],
        reverse=True,
    )
    if dated_dirs:
        # Check the most recent date first
        latest_date_dir = dated_dirs[0]
        cand = latest_date_dir / device / "raw"
        if cand.exists() and cand.is_dir():
            return cand

    # Fallback if no dated directory is found
    return FRAMES_BASE_DIR / device / "raw"

def _config_dict_for(cam: Camera, device: str, site: str, src_dir: Path) -> dict:
    blob_prefix = f"{AZURE_PREFIX_ROOT}/{site}/{device}".strip("/")

    video_output_base_dir = Path("/home/sentinel/Projects/ambient-machine/results/videos")
    date_str = dt.datetime.now().strftime('%Y-%m-%d')
    structured_archive_dir = video_output_base_dir / date_str / device
    return {
        "source_dirs": [str(src_dir)],
        "include_globs": _include_globs(cam),
        "exclude_globs": _exclude_globs(cam),
        "period_seconds": _period_of(cam),
        "align_to_hour": True,
        "lag_seconds": DEFAULT_LAG,
        "archive_dir": str(structured_archive_dir),
        "timezone": _timezone_of(cam),
        "azure": {
            "connection_string": AZURE_CONN,
            "container": AZURE_CONTAINER,
            "blob_prefix": blob_prefix,
        },
        "mongo": {
            "uri": MONGO_URI,
            "db_name": MONGO_DB,
            "alias": "default",
        },
        "delete_local_archive_after_upload": True,
        "dry_run": False,
    }

def _write_yaml(camera_name: str, site: str, cfg: dict) -> Path:
    target_dir = CONFIG_DIR / site
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{camera_name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path

def _ensure_dirs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def _log_file_for(site: str, camera_name: str) -> Path:
    return LOG_DIR / f"video_backup_{site}_{camera_name}.log"

# _ensure_cron is removed
# def _ensure_cron(...): ...

def _run_job_in_thread(yaml_path: Path, level: int) -> int:
    """
    Run backup_service for one YAML config inside the current process (thread).
    We import the module and call BackupJob.run_once() directly.
    """
    # Lazy import to keep this script's import-time light
    from src.services import backup_service as vb

    site = yaml_path.parent.name
    device = yaml_path.stem
    log_path = _log_file_for(site, device)
    logger = logging.getLogger(f"backup_service[{site}:{device}]")
    logger.setLevel(level)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    try:
        cfg = vb.load_config(str(yaml_path))
        vb.connect_mongo(cfg.mongo)
        job = vb.BackupJob(cfg, logger=logger)
        rc = job.run_once()
        return rc
    except Exception as e:
        logger.exception(f"Job failed for {yaml_path}: {e}")
        return 1
    finally:
        try:
            logger.removeHandler(fh)
            fh.close()
        except Exception:
            pass

# ---------- public API ----------

def generate_backup_configs(logger: logging.Logger) -> List[Path]:
    """
    Generates per-camera YAML configs. Returns a list of YAML paths created.
    """
    _ensure_dirs()
    written_yaml_paths: List[Path] = []

    _connect_mongo()
    cameras = _fetch_cameras()
    if not cameras:
        logger.info("No cameras found to configure (active=true & backup=true or missing).")
        return written_yaml_paths

    logger.info(f"Found {len(cameras)} camera(s) to configure.")
    for cam in cameras:
        device = _device_of(cam)
        if not device:
            logger.warning(f"Camera doc missing a device name: {cam.id}")
            continue
        site = _site_of(cam)
        src_dir = _source_dir_for(cam, device)
        cfg = _config_dict_for(cam, device, site, src_dir)
        yaml_path = _write_yaml(device, site, cfg)
        written_yaml_paths.append(yaml_path)
        logger.info(f"Generated config for '{device}' at: {yaml_path}")

    logger.info("Config generation complete.")
    return written_yaml_paths

# ---------- CLI entrypoint ----------

def _make_logger(name: str, level: str = "INFO", log_to_file: bool = False) -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_to_file:
        try:
            _ensure_dirs()
            # Note: Changed the log file name to avoid confusion with the old script name
            fh = logging.FileHandler(LOG_DIR / "video_backup_config_generator.log")
            fh.setLevel(lvl)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            logger.warning(f"Couldn't attach file logger: {e}")
    return logger

def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse
    # Updated the description to be more accurate
    parser = argparse.ArgumentParser(
        prog="video_backup_config_generator",
        description="Generate per-camera YAML configs. With --run-now, run backup jobs in parallel threads.",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"),
                        help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log-to-file", action="store_true",
                        help=f"Also write logs to {LOG_DIR}/video_backup_config_generator.log")
    parser.add_argument("--run-now", action="store_true",
                        help="After generating configs, immediately run each camera's backup job once in parallel.")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max concurrent threads for --run-now (default: #cameras).")
    parser.add_argument("--version", action="version", version="video_backup_config_generator 1.3")

    args = parser.parse_args(list(argv) if argv is not None else None)
    logger = _make_logger("video-backup-generator", args.log_level, args.log_to_file)

    logger.info(f"CONFIG_DIR: {CONFIG_DIR}")
    logger.info(f"LOG_DIR:    {LOG_DIR}")
    logger.info(f"BASE_DIR:   {FRAMES_BASE_DIR}")

    try:
        # Renamed the main function call
        yaml_paths = generate_backup_configs(logger)

        if args.run_now and yaml_paths:
            level = getattr(logging, args.log_level.upper(), logging.INFO)
            max_workers = args.max_parallel or len(yaml_paths)
            logger.info(f"[run-now] Starting {len(yaml_paths)} job(s) with up to {max_workers} thread(s)...")

            failures = 0
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vbjob") as pool:
                futs = {pool.submit(_run_job_in_thread, yp, level): yp for yp in yaml_paths}
                for fut in as_completed(futs):
                    yp = futs[fut]
                    try:
                        rc = fut.result()
                        if rc != 0:
                            failures += 1
                            logger.error(f"[run-now] Job exited with code {rc} for {yp.name}")
                        else:
                            logger.info(f"[run-now] Job completed successfully for {yp.name}")
                    except Exception as e:
                        failures += 1
                        logger.error(f"[run-now] Job raised an exception for {yp.name}: {e}", exc_info=True)

            if failures:
                logger.warning(f"[run-now] {failures} job(s) failed.")

        logger.info("Done.")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"A fatal error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())