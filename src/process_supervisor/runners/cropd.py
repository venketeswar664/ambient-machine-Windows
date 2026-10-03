from __future__ import annotations
import os, sys, logging
from pathlib import Path
from typing import Iterable
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from bson.objectid import ObjectId

THIS = Path(__file__).resolve()
SRC_DIR = THIS.parents[3]           # .../src
REPO_ROOT = SRC_DIR.parent
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.services.store_crop_service import CropStorageService, CronManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cropd")

def _iterate_devices() -> Iterable[ObjectId]:
    cm = CronManager(logger=log)
    ids = cm.get_distinct_device_ids() or []
    for did in ids:
        yield ObjectId(did) if not isinstance(did, ObjectId) else did

def run_once():
    svc = CropStorageService()
    cnt = 0
    for device_id in _iterate_devices():
        cnt += 1
        try:
            log.info(f"[crop] device={device_id} (last 10 min)")
            svc.run(minutes_ago=10, device_id=device_id)
        except Exception as e:
            log.exception(f"[crop] failed for device {device_id}: {e}")
    if cnt == 0:
        log.info("No devices found for cropping.")

def main():
    run_once()
    sched = BlockingScheduler()
    sched.add_job(run_once, IntervalTrigger(minutes=10), id="crop-interval", replace_existing=True)
    sched.start()

if __name__ == "__main__":
    main()
