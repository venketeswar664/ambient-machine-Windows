import os
import re
import cv2
from datetime import datetime

# ---------------- CONFIG ----------------
# ---------------- CONFIG ----------------
FRAME_DIR = "C:\\Users\\nj.camera\\indriya\\ambient-machine\\results\\frames\\2026-08-25\\C3\\plotted"

START_FRAME = "C:\\Users\\nj.camera\\indriya\\ambient-machine\\results\\frames\\2026-08-25\\C3\\plotted\\2026-08-25_14-31-31_741_f34871.jpg"

END_FRAME = "C:\\Users\\nj.camera\\indriya\\ambient-machine\\results\\frames\\2026-08-25\\C3\\plotted\\2026-08-25_16-29-52_685_f20418.jpg"

OUTPUT_VIDEO = "C:\\Users\\nj.camera\\Indriya\\ambient-machine\\videos\\seawoods.mp4"
FPS = 5

# ----------------------------------------


def extract_datetime_from_filename(path):
    """
    Extract datetime from filename like:
    2026-06-04_04-30-52_594_f2.jpg
    """

    fname = os.path.basename(path)

    match = re.search(
        r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_(\d+)",
        fname
    )

    if not match:
        return None

    date_part = match.group(1)
    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)
    millis = match.group(5)

    dt_str = f"{date_part} {hour}:{minute}:{second}.{millis}"

    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")


def get_all_frames_between(start_frame, end_frame):
    start_dt = extract_datetime_from_filename(start_frame)
    end_dt = extract_datetime_from_filename(end_frame)

    if start_dt is None:
        raise ValueError(f"Could not parse START_FRAME: {start_frame}")

    if end_dt is None:
        raise ValueError(f"Could not parse END_FRAME: {end_frame}")

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    selected_frames = []

    for fname in os.listdir(FRAME_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        frame_path = os.path.join(FRAME_DIR, fname)
        frame_dt = extract_datetime_from_filename(frame_path)

        if frame_dt is None:
            continue

        if start_dt <= frame_dt <= end_dt:
            selected_frames.append((frame_dt, frame_path))

    selected_frames.sort(key=lambda x: x[0])

    return [path for _, path in selected_frames]


def create_video(frame_paths, output_video, fps):
    if not frame_paths:
        print("No frames found in selected range.")
        return

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    first_frame = cv2.imread(frame_paths[0])

    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    count = 0

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        count += 1

    writer.release()

    print("Video saved:", output_video)
    print("Total frames used:", count)
    print("FPS:", fps)
    print("Duration seconds:", round(count / fps, 2))


if __name__ == "__main__":
    frames = get_all_frames_between(START_FRAME, END_FRAME)

    print("Frames found:", len(frames))

    if frames:
        print("First frame:", os.path.basename(frames[0]))
        print("Last frame :", os.path.basename(frames[-1]))

    create_video(frames, OUTPUT_VIDEO, FPS)