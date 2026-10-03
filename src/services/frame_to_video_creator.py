# Save this as src/services/frame_to_video_creator.py

import datetime as dt
import logging
from pathlib import Path
from collections import defaultdict
import cv2

# --- Configuration ---
# The parent directory where all frame folders are located
FRAMES_BASE_DIR = Path("/home/sentinel/Projects/ambient-machine/results/frames")
# The parent directory where all output videos will be saved
VIDEO_OUTPUT_DIR = Path("/home/sentinel/Projects/ambient-machine/results/videos")
# How far back to look for frames (in seconds). Should be greater than the cron schedule.
LOOKBACK_SECONDS = 360 # 6 minutes, to overlap with a 5-minute cron
FPS = 5

# (You can copy the create_video_from_frames function from backup_service.py here)
# ...

def create_videos_for_camera(camera_dir: Path, logger: logging.Logger):
    """Finds frames for a single camera and creates 1-minute videos."""
    now = dt.datetime.now(dt.timezone.utc)
    pstart = now - dt.timedelta(seconds=LOOKBACK_SECONDS)
    pend = now

    all_frames = []
    raw_dir = camera_dir / "raw"
    if not raw_dir.is_dir():
        return

    for f in raw_dir.rglob("*.jpg"):
        try:
            mtime = dt.datetime.fromtimestamp(f.stat().st_mtime, tz=dt.timezone.utc)
            if pstart <= mtime < pend:
                all_frames.append(f)
        except FileNotFoundError:
            continue

    if not all_frames:
        return # No new frames to process

    # Group frames by minute
    frames_by_minute = defaultdict(list)
    # ... (same filename parsing logic as before to group frames) ...

    # Create a video for each minute chunk
    for minute_key, frame_paths in frames_by_minute.items():
        date_str = dt.datetime.strptime(minute_key, "%Y%m%d_%H%M00").strftime('%Y-%m-%d')
        output_dir = VIDEO_OUTPUT_DIR / date_str / camera_dir.name
        video_filename = f"video_{minute_key}.mp4"
        video_path = output_dir / video_filename

        if video_path.exists():
            continue # Skip if video already exists

        create_video_from_frames(frame_paths, video_path, logger)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    logger = logging.getLogger("frame-to-video-creator")
    
    # Find all camera directories within the most recent date folder
    dated_dirs = sorted(FRAMES_BASE_DIR.glob("*"), key=os.path.getmtime, reverse=True)
    if not dated_dirs:
        return
        
    latest_date_dir = dated_dirs[0]
    for camera_dir in latest_date_dir.iterdir():
        if camera_dir.is_dir():
            logger.info(f"Processing camera: {camera_dir.name}")
            create_videos_for_camera(camera_dir, logger)

if __name__ == "__main__":
    main()