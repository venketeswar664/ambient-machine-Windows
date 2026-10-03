"""
repeat_entry_runner.py
======================
Nightly runner for the Repeat Entry Detection pipeline.

Execution order:
  1. Extract today's customer crops from MongoDB  →  repeat_entry/results/footfall/YYYY-MM-DD/
  2. Run repeat_entry/main.py (embedding extraction + matching + DB push) as a subprocess

Scheduling:
  Managed by circus via independent.circus.ini.
  Cron: 30 17 * * *  (UTC)  =  23:00 IST

Manual rerun for a different date:
  REPEAT_ENTRY_DATE=2026-06-17 python src/process_supervisor/runners/repeat_entry_runner.py

Manual rerun for a single device only:
  REPEAT_ENTRY_DATE=2026-06-26 REPEAT_ENTRY_DEVICE=69ba968fd51e7a4ac615a026 python src/process_supervisor/runners/repeat_entry_runner.py
"""
from __future__ import annotations

import os
import sys
import logging
import subprocess
import datetime
from pathlib import Path

import cv2
from pymongo import MongoClient
from bson.objectid import ObjectId

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[3]          # .../ambient-machine
REPEAT_ENTRY_DIR = REPO_ROOT / "repeat_entry"

for p in (str(REPO_ROOT), str(REPEAT_ENTRY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = REPO_ROOT / "logs" / "circus"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "repeat_entry_service.runner.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("repeat_entry_runner")

# ---------------------------------------------------------------------------
# Config (mirrored from repeat_entry/config.py to avoid circular imports)
# ---------------------------------------------------------------------------
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://seawoods:sentinelMongo%40123@localhost:27017/sentinel?authSource=sentinel&replicaSet=rs0&directConnection=true"
)
DB_NAME = "sentinel"

# Date to process — override with env var for manual reruns
_env_date = os.environ.get("REPEAT_ENTRY_DATE", "")
if _env_date:
    try:
        TARGET_DATE = datetime.datetime.strptime(_env_date, "%Y-%m-%d").date()
    except ValueError:
        TARGET_DATE = datetime.date.today()
else:
    TARGET_DATE = datetime.date.today()

TARGET_DATE_STR = TARGET_DATE.strftime("%Y-%m-%d")

# Where crops are saved:  repeat_entry/results/footfall/YYYY-MM-DD/<track_folder>/
FOOTFALL_ROOT = REPEAT_ENTRY_DIR / "results" / "footfall"
TODAY_FOOTFALL_DIR = FOOTFALL_ROOT / TARGET_DATE_STR

# Optional single-device filter — override with env var for targeted reruns
_env_device = os.environ.get("REPEAT_ENTRY_DEVICE", "").strip()
TARGET_DEVICE_ID: ObjectId | None = ObjectId(_env_device) if _env_device else None


# ---------------------------------------------------------------------------
# Step 1 — Extract customer crops for active devices (all, or a single one)
# ---------------------------------------------------------------------------
def _get_active_devices(db) -> list[ObjectId]:
    """Return device IDs to process.

    If REPEAT_ENTRY_DEVICE is set, returns only that one device (must be
    active).  Otherwise returns all active devices.
    """
    if TARGET_DEVICE_ID is not None:
        # Verify the requested device is actually active
        doc = db["cameras"].find_one(
            {"_id": TARGET_DEVICE_ID, "active": {"$ne": False}},
            {"_id": 1},
        )
        if doc is None:
            log.warning(
                f"Device {TARGET_DEVICE_ID} not found or not active — skipping."
            )
            return []
        log.info(f"Single-device mode: {TARGET_DEVICE_ID}")
        return [TARGET_DEVICE_ID]

    pipeline = [
        {"$match": {"active": {"$ne": False}}},
        {"$project": {"_id": 1}},
    ]
    docs = list(db["cameras"].aggregate(pipeline))
    ids = [d["_id"] for d in docs]
    log.info(f"Found {len(ids)} active device(s).")
    return ids


def extract_crops_for_today() -> None:
    """
    Extract customer crops from MongoDB for TARGET_DATE across all active devices.
    Saves images to: repeat_entry/results/footfall/YYYY-MM-DD/<timestamp_trackId>/<frame>.jpg
    """
    log.info(f"=== Step 1: Crop Extraction for {TARGET_DATE_STR} ===")

    start_dt = datetime.datetime.combine(TARGET_DATE, datetime.time(0, 0, 0))
    end_dt   = datetime.datetime.combine(TARGET_DATE, datetime.time(23, 59, 59, 999999))

    log.info(f"Date range: {start_dt} → {end_dt}")
    log.info(f"Output dir: {TODAY_FOOTFALL_DIR}")

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    try:
        db = client[DB_NAME]
        device_ids = _get_active_devices(db)

        if not device_ids:
            log.warning("No active devices found. Skipping crop extraction.")
            return

        total_crops_saved = 0

        for device_id in device_ids:
            log.info(f"--- Processing device: {device_id} ---")

            # ── Fetch valid customer track IDs ──────────────────────────────
            track_query = {
                "start_time": {"$gte": start_dt, "$lt": end_dt},
                "device": device_id,
                "label": "customer",
            }
            valid_tracks: dict[str, str] = {}
            for doc in db.track_id_metadata.find(track_query):
                tid = doc.get("track_id")
                start_time_obj = doc.get("start_time")
                if tid and start_time_obj:
                    valid_tracks[tid] = start_time_obj.strftime("%Y-%m-%d_%H-%M-%S")

            log.info(f"  Valid customer tracks: {len(valid_tracks)}")
            if not valid_tracks:
                continue

            # ── Fetch metadata frames ───────────────────────────────────────
            metadata_query = {
                "device": device_id,
                "time_stamp": {
                    "$gte": start_dt - datetime.timedelta(minutes=5),
                    "$lt":  end_dt   + datetime.timedelta(minutes=5),
                },
                "track_ids_info": {"$exists": True, "$ne": {}},
            }
            total_docs = db.metadata.count_documents(metadata_query)
            log.info(f"  Metadata docs to scan: {total_docs}")

            cursor = db.metadata.find(metadata_query).sort("time_stamp", 1)
            saved = 0
            processed = 0

            for doc in cursor:
                processed += 1
                if processed % 5000 == 0:
                    log.info(f"  Progress: {processed}/{total_docs} | saved: {saved}")

                raw_frame_path = doc.get("raw_frame_path")
                if not raw_frame_path:
                    continue
                if not os.path.isabs(raw_frame_path):
                    raw_frame_path = os.path.join(str(REPO_ROOT), raw_frame_path.lstrip("./"))
                if not os.path.exists(raw_frame_path):
                    continue

                track_ids_dict = doc.get("track_ids_info", {})
                timestamp_obj  = doc.get("time_stamp")
                if not track_ids_dict or not timestamp_obj:
                    continue

                date_str = timestamp_obj.strftime("%Y-%m-%d")
                frame = None

                for track_id, info in track_ids_dict.items():
                    if track_id not in valid_tracks:
                        continue
                    bbox = info.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue

                    first_seen_str   = valid_tracks[track_id]
                    track_folder     = f"{first_seen_str}_{track_id}"
                    track_dir        = TODAY_FOOTFALL_DIR / track_folder
                    track_dir.mkdir(parents=True, exist_ok=True)

                    ts_ms   = timestamp_obj.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                    out_path = track_dir / f"{ts_ms}-customer.jpg"
                    if out_path.exists():
                        continue

                    if frame is None:
                        frame = cv2.imread(raw_frame_path)
                        if frame is None:
                            break

                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = max(0, x2), max(0, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    if cv2.imwrite(str(out_path), crop):
                        saved += 1

            total_crops_saved += saved
            log.info(f"  Device {device_id}: {saved} crops saved.")

        log.info(f"=== Step 1 Complete: {total_crops_saved} total crops saved ===")

    finally:
        client.close()


# ---------------------------------------------------------------------------
# Step 2 — Run repeat_entry/main.py (embedding extraction + matching + DB push)
# ---------------------------------------------------------------------------
def run_repeat_entry_pipeline() -> None:
    """
    Spawns repeat_entry/main.py as a child process.
    Keeping it as a subprocess isolates GPU memory cleanly.
    """
    log.info(f"=== Step 2: Repeat Entry Pipeline (date={TARGET_DATE_STR}) ===")

    python_bin = sys.executable
    script     = str(REPEAT_ENTRY_DIR / "main.py")

    env = dict(os.environ)
    env["REPEAT_ENTRY_DATE"] = TARGET_DATE_STR
    # Ensure repeat_entry is importable
    env["PYTHONPATH"] = f"{REPEAT_ENTRY_DIR}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    cmd = [python_bin, "-u", script]
    log.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env, cwd=str(REPEAT_ENTRY_DIR), capture_output=False)

    if result.returncode == 0:
        log.info("=== Step 2 Complete: Pipeline finished successfully ===")
    else:
        log.error(f"=== Step 2 FAILED: exit code {result.returncode} ===")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info(f"🌙 Repeat Entry Nightly Runner — {TARGET_DATE_STR}")
    log.info("=" * 60)

    extract_crops_for_today()
    run_repeat_entry_pipeline()

    log.info("✅ Nightly run complete.")


if __name__ == "__main__":
    main()
