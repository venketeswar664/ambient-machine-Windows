"""
reprocess_date.py
=================
Reprocesses all raw frames for a given date exactly the same way the live
pipeline (CameraOrchestratorEngine / run_camera_pipeline) does:

  1. Fetch CameraConfig from the database.
  2. Find the first service that has a pipelinePath.
  3. Load the JSON at pipelinePath to get the base detector_config.
  4. Inject deviceName, ip_camera, and zones into the config.
  5. Override model settings from the DB (baseModel on the Cameras doc).
  6. Initialise the Detector class named in the JSON.
  7. Feed every raw .jpg from results/frames/<date>/<cam>/raw/ through the
     Detector in chronological order, using the original timestamp embedded
     in the filename.

Usage:
    python3 reprocess_date.py 2026-03-23
"""

import sys
import os
import json
import time
import datetime
import glob
import logging
import numpy as np
import cv2

sys.path.append(os.path.abspath("."))

from src.database.database import Database
from src.database.schemas.cameras_schema import Cameras
from src.database.schemas.models_schema import Models
from src.database.schemas.camera_config_schema import CameraConfig


# ---------------------------------------------------------------------------
# Minimal mock that satisfies Detector.__init__() without a live RTSP feed
# ---------------------------------------------------------------------------
class _MockCapture:
    """Thin stub for cv2.VideoCapture — returns a black frame on .read()."""
    def __init__(self):
        self._blank = np.zeros((720, 1280, 3), dtype=np.uint8)

    def read(self):
        return True, self._blank.copy()

    def release(self):
        pass

    def isOpened(self):
        return True


class MockCamera:
    """
    Mimics the interface of IP_Camera without opening any network connection.
    Detector only reads: is_open, is_video_file, fps, frame_width,
    frame_height, rotation, capture.read(), ip_address.
    """
    def __init__(self, ip_address: str, device_name: str, rotation: int = 0):
        self.ip_address  = ip_address
        self.device_name = device_name
        self.rotation    = rotation
        self.fps         = 5.0
        self.is_open     = True
        self.is_video_file = True   # suppresses GStreamer attempts
        self.frame_width  = 1280
        self.frame_height = 720
        self.capture      = _MockCapture()

    def close(self):
        self.is_open = False


# ---------------------------------------------------------------------------
# Core reprocessing logic for ONE camera
# ---------------------------------------------------------------------------
def process_camera_frames(
    date_str: str,
    camera_name: str,
    native_width: int = None,
    start_time: str = None,
    end_time: str = None,
    start_frame_idx: int = 0,
):
    raw_dir = f"./results/frames/{date_str}/{camera_name}/raw/"
    if not os.path.isdir(raw_dir):
        print(f"[{camera_name}] No raw directory found: {raw_dir}")
        return

    frames = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    if not frames:
        print(f"[{camera_name}] No frames found in {raw_dir}")
        return

    # --- Time-range filter ---
    # Parse --start-time / --end-time (HH:MM:SS, UTC) and keep only frames
    # whose embedded timestamp falls inside [start_time, end_time].
    def _parse_frame_utc(fpath: str):
        """Return the UTC datetime embedded in the filename, or None."""
        bn = os.path.basename(fpath).replace(".jpg", "")
        parts = bn.rsplit("_f", 1)
        if len(parts) != 2:
            return None
        try:
            return datetime.datetime.strptime(parts[0], "%Y-%m-%d_%H-%M-%S_%f").replace(
                tzinfo=datetime.timezone.utc
            )
        except ValueError:
            return None

    if start_time or end_time:
        _date = datetime.date.fromisoformat(date_str)
        _tz   = datetime.timezone.utc

        def _make_dt(t_str):
            h, m, s = (int(x) for x in t_str.split(":"))
            return datetime.datetime(_date.year, _date.month, _date.day, h, m, s, tzinfo=_tz)

        _start_dt = _make_dt(start_time) if start_time else None
        _end_dt   = _make_dt(end_time)   if end_time   else None

        before = len(frames)
        filtered = []
        for fp in frames:
            ft = _parse_frame_utc(fp)
            if ft is None:
                continue
            if _start_dt and ft < _start_dt:
                continue
            if _end_dt and ft > _end_dt:
                continue
            filtered.append(fp)
        frames = filtered
        print(
            f"[{camera_name}] Time filter {start_time or '*'}–{end_time or '*'} UTC: "
            f"{len(frames)}/{before} frames kept."
        )



    # ------------------------------------------------------------------
    # 1. Fetch camera document
    # ------------------------------------------------------------------
    cam_doc = Cameras.objects(deviceName=camera_name).first()
    if not cam_doc:
        print(f"[{camera_name}] Camera not found in DB — skipping.")
        return

    # Auto-resume logic
    if start_frame_idx == 0:
        try:
            from src.database.schemas.metadata_schema import Metadata
            meta_cls = Metadata
            if "MetadataRcp" in globals():
                meta_cls = globals()["MetadataRcp"]
            
            _date_obj = datetime.date.fromisoformat(date_str)
            _start_dt = datetime.datetime(_date_obj.year, _date_obj.month, _date_obj.day, tzinfo=datetime.timezone.utc)
            _end_dt = _start_dt + datetime.timedelta(days=1)
            
            last_meta = meta_cls.objects(
                device=cam_doc.id,
                time_stamp__gte=_start_dt,
                time_stamp__lt=_end_dt
            ).order_by("-time_stamp").first()
            
            if last_meta and last_meta.raw_frame_path:
                last_basename = os.path.basename(last_meta.raw_frame_path)
                idx = next((i for i, f in enumerate(frames) if os.path.basename(f) == last_basename), -1)
                if idx != -1:
                    start_frame_idx = idx + 1
                    print(f"[{camera_name}] Auto-resuming from DB. Last processed frame was {last_basename}.")
        except Exception as e:
            print(f"[{camera_name}] Auto-resume check failed: {e}")

    if start_frame_idx > 0:
        if start_frame_idx < len(frames):
            frames = frames[start_frame_idx:]
            print(f"[{camera_name}] Skipping first {start_frame_idx} frames. {len(frames)} remaining.")
        else:
            print(f"[{camera_name}] start_frame_idx ({start_frame_idx}) >= total frames ({len(frames)}). Skipping all.")
            return

    if not frames:
        print(f"[{camera_name}] No frames in the requested time range — skipping.")
        return

    print(f"[{camera_name}] Found {len(frames)} frames. Initialising pipeline…")

    # ------------------------------------------------------------------
    # 2. Fetch CameraConfig and pick the first service with a pipelinePath
    # ------------------------------------------------------------------
    cam_config = CameraConfig.objects(cameraOid=cam_doc.id).order_by("-timeStamp").first()
    if not cam_config or not cam_config.services:
        print(f"[{camera_name}] No CameraConfig / services found — skipping.")
        return

    pipeline_service_item = None
    for svc_item in cam_config.services:
        svc = svc_item.service
        if svc and getattr(svc, "pipelinePath", None):
            pipeline_service_item = svc_item
            break

    if pipeline_service_item is None:
        print(f"[{camera_name}] No service with a pipelinePath — skipping.")
        return

    service = pipeline_service_item.service
    zones   = [zone.id for zone in pipeline_service_item.zones]

    print(f"[{camera_name}] Using pipelinePath: {service.pipelinePath}")
    print(f"[{camera_name}] Zones: {zones}")

    # ------------------------------------------------------------------
    # 3. Load JSON pipeline config  (mirrors CameraOrchestratorEngine)
    # ------------------------------------------------------------------
    try:
        with open(service.pipelinePath, "r") as f:
            pipeline_cfg = json.load(f)
    except Exception as e:
        print(f"[{camera_name}] Failed to load pipelinePath JSON: {e}")
        return

    detector_config = pipeline_cfg["pipeline"]["detector"]["config"]
    module_path     = pipeline_cfg["pipeline"]["detector"]["module"]
    class_name      = pipeline_cfg["pipeline"]["detector"]["name"]

    # ------------------------------------------------------------------
    # 4. Inject camera-specific fields (exact mirror of run_camera_pipeline)
    # ------------------------------------------------------------------
    rotation = getattr(cam_doc, "rotation", 0) or 0
    detector_config["deviceName"] = camera_name
    detector_config["zones"]      = zones
    detector_config["ip_camera"]  = MockCamera(
        ip_address=getattr(cam_doc, "cameraAddress", "mock://") or "mock://",
        device_name=camera_name,
        rotation=rotation,
    )

    # ------------------------------------------------------------------
    # 5. Override model from DB  (uses baseModel on the Cameras doc)
    # ------------------------------------------------------------------
    model_ref = getattr(cam_doc, "baseModel", None)
    model_doc = None
    if model_ref:
        if hasattr(model_ref, "modelName"):
            model_doc = model_ref  # already dereferenced
        else:
            model_doc = Models.objects(id=model_ref).first()

    if model_doc:
        detector_config.update({
            "modelName":     model_doc.modelName,
            "modelPath":     model_doc.modelPath,
            "modelType":     getattr(model_doc, "modelType", None),
            "conf":          model_doc.conf,
            "classIds":      model_doc.classIds,
            "trackerPath":   model_doc.trackerPath,
            "convertEngine": model_doc.convertEngine,
            "inputDims":     model_doc.inputDims,
        })
        print(f"[{camera_name}] Model override from DB: {model_doc.modelName}")
    else:
        print(f"[{camera_name}] No model found in DB — using JSON defaults.")

    # Always save at 1280 px width (same as live pipeline default)
    detector_config.setdefault("save_max_width", 1280)

    # ------------------------------------------------------------------
    # 6. Dynamically import and instantiate the Detector class
    # ------------------------------------------------------------------
    logger = logging.getLogger(camera_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    try:
        mod = __import__(
            module_path.replace("/", ".").replace(".py", ""),
            fromlist=[class_name]
        )
        detector_class = getattr(mod, class_name)
    except Exception as e:
        print(f"[{camera_name}] Cannot import detector '{module_path}.{class_name}': {e}")
        return

    try:
        detector = detector_class(config=detector_config, logger=logger)
    except Exception as e:
        import traceback
        print(f"[{camera_name}] Detector __init__ failed: {e}")
        traceback.print_exc()
        return

    # Disable in-detector rotation: frames on disk are already rotated
    detector.rotation = 0

    # Save plotted frames to the external drive, keep raw as it is
    if hasattr(detector, "frame_handler"):
        detector.frame_handler.base_dir = "C:\\Users\\nj.camera\\indriya\\ambient-machine\\results\\frames" 

    # ------------------------------------------------------------------
    # 6b. Scale zone geometries from native resolution → saved frame size
    #
    # Zones in the DB are drawn at native_width (default 2560px / 2560x1440).
    # Frames on disk are saved at 1280x720.  Scale = 1280/2560 = 0.5.
    # ------------------------------------------------------------------
    if native_width:
        _probe_frame = cv2.imread(frames[0]) if frames else None
        disk_width = _probe_frame.shape[1] if _probe_frame is not None else 1280
        if disk_width < native_width:
            zone_scale = disk_width / native_width
            print(f"[{camera_name}] Scaling zones ×{zone_scale:.4f} "
                  f"({native_width}px → {disk_width}px)")
            from shapely.affinity import scale as shapely_scale
            for zname, zinfo in detector.zone_manager.zone_dict.items():
                zinfo["shape"] = shapely_scale(
                    zinfo["shape"],
                    xfact=zone_scale,
                    yfact=zone_scale,
                    origin=(0, 0),
                )
        else:
            print(f"[{camera_name}] Frames are {disk_width}px — zone scaling not required.")

    # ------------------------------------------------------------------
    # 7. Feed every raw frame through the detector
    # ------------------------------------------------------------------
    processed_count = 0
    total = len(frames)

    for fpath in frames:
        basename = os.path.basename(fpath)
        parts    = basename.replace(".jpg", "").split("_")

        # Expected format: YYYY-MM-DD_HH-MM-SS_mmm_f<num>.jpg
        if len(parts) >= 4:
            date_part     = parts[0]
            time_part     = parts[1].replace("-", ":")
            ms_part       = parts[2]
            frame_num_str = parts[3].replace("f", "")
            dt_str = f"{date_part} {time_part}.{ms_part}"
            try:
                ts_utc = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                ts_utc = ts_utc.replace(tzinfo=datetime.timezone.utc)
                frame_num = int(frame_num_str)
            except ValueError:
                ts_utc    = datetime.datetime.now(datetime.timezone.utc)
                frame_num = 0
        else:
            ts_utc    = datetime.datetime.now(datetime.timezone.utc)
            frame_num = 0

        frame = cv2.imread(fpath)
        if frame is None:
            print(f"[{camera_name}] Could not read frame: {fpath} — skipping.")
            continue

        try:
            detector.predict(frame, ts_utc, frame_num, raw_path=fpath)
        except Exception as e:
            import traceback
            print(f"[{camera_name}] predict() failed on {fpath}: {e}")
            traceback.print_exc()
            continue

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"[{camera_name}] Processed {processed_count}/{total} frames…")

    # ------------------------------------------------------------------
    # 8. Flush & clean up
    # ------------------------------------------------------------------
    print(f"[{camera_name}] Done ({processed_count}/{total}). Flushing…")
    time.sleep(5)

    try:
        detector.frame_handler.close()
    except Exception:
        pass
    try:
        detector.metadata_handler.close()
    except Exception:
        pass

    print(f"[{camera_name}] Cleanup complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Reprocess raw frames for a given date through the live pipeline."
    )
    parser.add_argument("date", help="Date to reprocess in YYYY-MM-DD format")
    parser.add_argument(
        "--native-width",
        type=int,
        default=2560,
        metavar="PX",
        help="Native camera frame width (px) at which zones were drawn in the DB. Default: 2560.",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        metavar="CAM",
        help="Skip all cameras that sort before this name (e.g. C14). Useful for resuming.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        metavar="CAM",
        help="Process only this specific camera (e.g. C1).",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        metavar="HH:MM:SS",
        help="Only process frames at or after this UTC time (e.g. 16:06:11).",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        metavar="HH:MM:SS",
        help="Only process frames at or before this UTC time (e.g. 16:30:00).",
    )
    parser.add_argument(
        "--start-frame-idx",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N frames (useful for resuming a terminated job).",
    )
    args = parser.parse_args()

    date_str     = args.date
    native_width = args.native_width

    db = Database()
    db.connect_db()

    base_dir = f"./results/frames/{date_str}"
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        sys.exit(1)

    camera_names = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    camera_names.sort()

    if not camera_names:
        print(f"No camera directories found under {base_dir}")
        sys.exit(0)

    # Apply --camera filter
    if args.camera:
        if args.camera in camera_names:
            camera_names = [args.camera]
        else:
            print(f"Camera '{args.camera}' not found in {base_dir}")
            sys.exit(0)

    # Apply --start-from filter
    start_from = args.start_from
    if start_from and not args.camera:
        skipped = [c for c in camera_names if c < start_from]
        camera_names = [c for c in camera_names if c >= start_from]
        if skipped:
            print(f"Skipping {len(skipped)} camera(s) (already done): {skipped}")
        if not camera_names:
            print(f"No cameras found at or after '{start_from}' — nothing to do.")
            sys.exit(0)

    print(f"Cameras to process: {camera_names}")
    if native_width:
        print(f"Native camera width override: {native_width}px")

    for cam_name in camera_names:
        print(f"\n{'='*50}")
        print(f" Processing: {cam_name}  |  Date: {date_str}")
        print(f"{'='*50}")
        process_camera_frames(
            date_str,
            cam_name,
            native_width=native_width,
            start_time=args.start_time,
            end_time=args.end_time,
            start_frame_idx=args.start_frame_idx,
        )

    print("\nAll cameras processed.")


if __name__ == "__main__":
    main()
