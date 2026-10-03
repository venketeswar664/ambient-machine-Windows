import os
import tempfile
import shutil
import subprocess
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.son import SON
from pathlib import Path
import schedule
import time
import traceback

MONGODB_URI = "mongodb://sentinel:sentinelMongo_test123@10.8.0.26:27018/sentinel_warehouse?directConnection=true"
DB_NAME = "sentinel_warehouse"
COLLECTION_NAME = "nexus_test"

DESKTOP_PATH = "/home/sentinel/Projects/ambient-machine/"

# Cron schedule: Run every 1 minute
CRON_SCHEDULE_MINUTES = 1


# -----------------------------
# Helper : Get UTC day bounds
# -----------------------------
def get_utc_day_bounds():
    now = datetime.utcnow()
    start = datetime(now.year, now.month, now.day)
    end = start + timedelta(hours=23, minutes=59, seconds=59, milliseconds=999)
    return start, end


# -----------------------------
# MongoDB Aggregation Pipeline
# -----------------------------
def evidence_pipeline(start_day, end_day):
    return [
        {
            "$match": {
                "$expr": {
                    "$and": [
                        {"$gte": ["$startTime", datetime(2025, 11, 12)]},
                        {"$lte": ["$endTime", datetime(2025, 11, 13)]},
                        {
                            "$gt": [
                                {
                                    "$dateDiff": {
                                        "startDate": "$endTime",
                                        "endDate": datetime.utcnow(),
                                        "unit": "minute",
                                    }
                                },
                                5,
                            ]
                        },
                    ]
                },
                "evidencePath": {"$exists": False},
                "processingError": {"$exists": False},  # Don't reprocess failed items
                "keyword": "warehouseEntry",
            }
        },
        {
            "$lookup": {
                "from": "metadata",
                "let": {
                    "startTime": "$startTime",
                    "endTime": "$endTime",
                    "trackID": "$trackID",
                },
                "localField": "device",
                "foreignField": "device",
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$gte": ["$time_stamp", "$$startTime"]},
                                    {"$lte": ["$time_stamp", "$$endTime"]},
                                ]
                            }
                        }
                    },
                    {"$sort": SON([("time_stamp", 1)])},
                    {
                        "$project": {
                            "_id": 0,
                            "rawFramePath": "$raw_frame_path",
                            "bbox": {
                                "$getField": {
                                    "field": "$$trackID",
                                    "input": "$track_ids_info",
                                }
                            },
                        }
                    },
                ],
                "as": "frames",
            }
        },
        {"$project": {"_id": 1, "device": 1, "trackID": 1, "frames": 1}},
    ]


# ----------------------------------------
# Run FFmpeg: apply bounding boxes frame-wise
# ----------------------------------------
def create_video_with_bboxes(frames, output_video_path, evidence_id):
    """
    Create video with bounding boxes, handling corrupted frames gracefully
    Returns: (success: bool, error_log: dict)
    """
    temp_dir = tempfile.mkdtemp(prefix="ffmpeg_frames_")
    print(f"Temp directory created: {temp_dir}")
    
    error_log = {
        "corrupted_frames": [],
        "missing_frames": [],
        "invalid_bbox_frames": [],
        "low_confidence_frames": [],
        "successful_frames": 0,
        "total_frames": len(frames)
    }

    try:
        # Step 1 — Copy frames locally with error handling
        local_paths = []
        valid_frame_indices = []
        
        for idx, frame in enumerate(frames):
            try:
                remote = os.path.join(DESKTOP_PATH, frame["rawFramePath"])
                
                # Check if file exists
                if not os.path.exists(remote):
                    print(f"⚠️  Frame {idx}: File not found - {remote}")
                    error_log["missing_frames"].append({
                        "index": idx,
                        "path": frame["rawFramePath"],
                        "reason": "file_not_found"
                    })
                    continue
                
                ext = Path(remote).suffix
                local_path = os.path.join(temp_dir, f"frame_{idx:06d}{ext}")
                
                # Try to copy and verify the frame
                shutil.copyfile(remote, local_path)
                
                # Verify file is not corrupted (basic check)
                if os.path.getsize(local_path) == 0:
                    print(f"⚠️  Frame {idx}: Corrupted (0 bytes)")
                    error_log["corrupted_frames"].append({
                        "index": idx,
                        "path": frame["rawFramePath"],
                        "reason": "zero_bytes"
                    })
                    os.remove(local_path)
                    continue
                
                local_paths.append(local_path)
                valid_frame_indices.append(idx)
                error_log["successful_frames"] += 1
                
            except (IOError, OSError, PermissionError) as e:
                print(f"⚠️  Frame {idx}: Copy failed - {str(e)}")
                error_log["corrupted_frames"].append({
                    "index": idx,
                    "path": frame["rawFramePath"],
                    "reason": str(e)
                })
                continue
        
        # Check if we have enough valid frames
        if len(local_paths) == 0:
            raise ValueError("No valid frames to process")
        
        if len(local_paths) < len(frames) * 0.5:  # Less than 50% valid
            print(f"⚠️  Warning: Only {len(local_paths)}/{len(frames)} frames are valid")

        # Step 2 — Create concat file with only valid frames
        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in local_paths:
                f.write(f"file '{p}'\n")

        # Step 3 — Build ffmpeg drawbox filters for valid frames only
        filters = []
        for local_idx, original_idx in enumerate(valid_frame_indices):
            frame = frames[original_idx]
            bbox_data = frame.get("bbox")
            
            # Skip if bbox_data is missing or None
            if not bbox_data:
                error_log["invalid_bbox_frames"].append({
                    "index": original_idx,
                    "reason": "bbox_data_missing"
                })
                continue
            
            # Extract nested bbox coordinates and metadata
            box = bbox_data.get("bbox")
            confidence = bbox_data.get("confidence", 0)
            label_name = bbox_data.get("label_name", "unknown")

            # Skip frames with invalid bbox
            if not box or len(box) != 4:
                error_log["invalid_bbox_frames"].append({
                    "index": original_idx,
                    "reason": "invalid_bbox_format",
                    "bbox": box
                })
                continue
            
            # Optional: Add confidence threshold
            if confidence < 0.5:
                error_log["low_confidence_frames"].append({
                    "index": original_idx,
                    "confidence": confidence,
                    "label": label_name
                })
                continue

            try:
                x1, y1, x2, y2 = box
            except (ValueError, TypeError) as e:
                error_log["invalid_bbox_frames"].append({
                    "index": original_idx,
                    "reason": f"bbox_parse_error: {str(e)}",
                    "bbox": box
                })
                continue

            w, h = x2 - x1, y2 - y1

            # Ensure positive width and height
            if w <= 0 or h <= 0:
                error_log["invalid_bbox_frames"].append({
                    "index": original_idx,
                    "reason": "invalid_dimensions",
                    "width": w,
                    "height": h
                })
                continue

            # Use local_idx for the filter (sequential frame number in output video)
            filters.append(
                f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color=red:thickness=7:enable='eq(n,{local_idx})'"
            )

        filter_str = ",".join(filters) if filters else None

        # Step 4 — Run FFmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-r", "30",
            "-i", concat_list,
        ]

        if filter_str:
            ffmpeg_cmd += ["-vf", filter_str]

        ffmpeg_cmd += [
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            output_video_path
        ]

        print(f"Running FFmpeg with {len(local_paths)} valid frames...")
        result = subprocess.run(
            ffmpeg_cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        
        return True, error_log

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e.stderr}")
        error_log["ffmpeg_error"] = e.stderr
        return False, error_log
    
    except Exception as e:
        print(f"❌ Video creation failed: {str(e)}")
        error_log["general_error"] = str(e)
        return False, error_log
    
    finally:
        print("Cleaning temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------
# Main Evidence Creation Logic
# -----------------------------
def evidence_creation():
    """Main function to process evidence - called by scheduler"""
    print(f"\n{'='*60}")
    print(f"Running evidence creation job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    client = MongoClient(MONGODB_URI, maxPoolSize=20)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    try:
        start_day, end_day = get_utc_day_bounds()
        pipeline = evidence_pipeline(start_day, end_day)

        evidence_list = list(collection.aggregate(pipeline))
        print(f"Found {len(evidence_list)} evidence items to process\n")

        if len(evidence_list) == 0:
            print("✓ No new evidence to process")
            return

        processed_count = 0
        failed_count = 0

        for evidence in evidence_list:
            try:
                device = evidence["device"]
                frames = evidence["frames"]
                evidence_id = evidence["_id"]

                print(f"\n--- Processing device: {device} (ID: {evidence_id}) ---")

                if not frames:
                    print(f"⚠️  No frames for device {device}, marking as error")
                    collection.update_one(
                        {"_id": evidence_id},
                        {
                            "$set": {
                                "processingError": "no_frames",
                                "errorTimestamp": datetime.utcnow(),
                                "errorDetails": "No frames found in metadata"
                            }
                        }
                    )
                    failed_count += 1
                    continue

                # Updated output directory to the desired path
                output_dir = "/home/sentinel/Projects/ambient-machine/video_evidence/error_demo"
                os.makedirs(output_dir, exist_ok=True)

                filename = f"evidence_{device}_{int(datetime.utcnow().timestamp())}.mp4"
                output_path = os.path.join(output_dir, filename)

                # Create video with error handling
                success, error_log = create_video_with_bboxes(frames, output_path, evidence_id)

                if success:
                    print(f"\n✔ Evidence video created: {filename}")
                    print(f"  Total frames: {error_log['total_frames']}")
                    print(f"  Successful frames: {error_log['successful_frames']}")
                    print(f"  Corrupted frames: {len(error_log['corrupted_frames'])}")
                    print(f"  Missing frames: {len(error_log['missing_frames'])}")
                    print(f"  Invalid bbox frames: {len(error_log['invalid_bbox_frames'])}")
                    print(f"  Low confidence frames: {len(error_log['low_confidence_frames'])}")

                    # Update DB with evidence path and error log
                    update_data = {
                        "evidencePath": filename,
                        "processedTimestamp": datetime.utcnow(),
                        "frameErrorLog": error_log
                    }
                    
                    collection.update_one(
                        {"_id": evidence_id},
                        {"$set": update_data}
                    )
                    processed_count += 1
                    
                else:
                    print(f"\n❌ Failed to create video for device {device}")
                    
                    # Mark as processing error in DB
                    collection.update_one(
                        {"_id": evidence_id},
                        {
                            "$set": {
                                "processingError": "video_creation_failed",
                                "errorTimestamp": datetime.utcnow(),
                                "errorDetails": error_log
                            }
                        }
                    )
                    failed_count += 1

            except Exception as e:
                print(f"\n❌ Error processing device {evidence.get('device')}: {str(e)}")
                print(traceback.format_exc())
                
                # Mark as processing error
                try:
                    collection.update_one(
                        {"_id": evidence.get("_id")},
                        {
                            "$set": {
                                "processingError": "exception",
                                "errorTimestamp": datetime.utcnow(),
                                "errorDetails": {
                                    "error": str(e),
                                    "traceback": traceback.format_exc()
                                }
                            }
                        }
                    )
                except:
                    pass
                
                failed_count += 1

        print(f"\n{'='*60}")
        print(f"Job completed:")
        print(f"  ✔ Processed: {processed_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ Error in evidence creation: {str(e)}")
        print(traceback.format_exc())
    finally:
        client.close()


# -----------------------------
# Scheduler Setup
# -----------------------------
def run_scheduler():
    """Run the scheduler continuously"""
    print(f"\n{'='*60}")
    print(f"Evidence Creation Scheduler Started")
    print(f"{'='*60}")
    print(f"Schedule: Every {CRON_SCHEDULE_MINUTES} minute(s)")
    print(f"Database: {DB_NAME}.{COLLECTION_NAME}")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    # Schedule the job
    schedule.every(CRON_SCHEDULE_MINUTES).minutes.do(evidence_creation)
    
    # Run immediately on start
    evidence_creation()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    try:
        run_scheduler()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Scheduler stopped by user")
        print("="*60)
    except Exception as e:
        print("\n\n" + "="*60)
        print(f"Scheduler crashed: {e}")
        print(traceback.format_exc())
        print("="*60)