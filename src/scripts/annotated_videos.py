# """
# annotate_videos.py - ENHANCED VERSION WITH ROI ZONES

# This version handles various frame numbering schemes and adds ROI zone plotting from camera_zone_mapping collection.

# Features:
# - Plots bounding boxes from metadata
# - Plots ROI zones from camera zone mapping
# - Handles various frame numbering schemes
# - Robust error handling

# Usage example:
#     python annotate_videos.py \
#       --start-time "2025-06-13T12:00:00" \
#       --end-time   "2025-06-13T12:10:00" \
#       --output-dir "./results/plotted_videos" \
#       --mongo-uri "mongodb://user:pass@host:27017/mydb" \
#       --db-name   "mydb"
# """

# import os
# import argparse
# import datetime
# from urllib.parse import urlparse
# from dateutil import parser as date_parser

# from dotenv import load_dotenv
# import mongoengine as me
# from pymongo import MongoClient
# from bson import ObjectId
# import cv2

# def safe_int_conversion(value):
#     """Safely convert various frame number formats to integers"""
#     if isinstance(value, int):
#         return value
#     elif isinstance(value, str):
#         try:
#             # Try direct conversion first
#             return int(value)
#         except ValueError:
#             try:
#                 # Try hexadecimal conversion
#                 return int(value, 16)
#             except ValueError:
#                 # If all else fails, use hash of the string for consistent ordering
#                 return hash(value) % (2**31)
#     else:
#         # For other types (like ObjectId), convert to string then hash
#         return hash(str(value)) % (2**31)

# def get_camera_roi_zones(client, db_name, camera_name="Camera-018"):
#     """
#     Get ROI zones for a camera by matching cameras collection with camera_zone_mapping
#     ROI coordinates are directly in the camera_zone_mapping collection
    
#     Returns:
#         list: List of ROI zones with coordinates
#     """
#     db = client[db_name]
#     cameras_coll = db["cameras"]
#     zone_mapping_coll = db["camera_zone_mapping"]
    
#     print(f"\n=== FETCHING ROI ZONES FOR {camera_name} ===")
    
#     # Find camera document by device_name
#     camera_doc = cameras_coll.find_one({"device_name": camera_name})
#     if not camera_doc:
#         print(f"[WARN] Camera {camera_name} not found in cameras collection")
#         return []
    
#     camera_object_id = camera_doc["_id"]
#     print(f"Found camera with ObjectId: {camera_object_id}")
    
#     # Find zone mappings for this camera
#     zone_mappings = zone_mapping_coll.find({"camera": camera_object_id})
    
#     roi_zones = []
#     for mapping in zone_mappings:
#         print(f"Found zone mapping: {mapping['_id']}")
        
#         # Extract ROI directly from the mapping document
#         if "roi" in mapping:
#             roi_coordinates = mapping["roi"]
#             print(f"  Found ROI coordinates: {roi_coordinates}")
            
#             roi_zones.append({
#                 "mapping_id": str(mapping["_id"]),
#                 "coordinates": roi_coordinates
#             })
#         else:
#             print(f"  [WARN] No ROI found in mapping {mapping['_id']}")
        
#         # Also check if there are zones array with individual ROIs
#         if "zones" in mapping and isinstance(mapping["zones"], list):
#             for i, zone_data in enumerate(mapping["zones"]):
#                 print(f"  Checking zone {i}: {zone_data}")
                
#                 # If zone_data is a dict with ROI
#                 if isinstance(zone_data, dict) and "roi" in zone_data:
#                     roi_zones.append({
#                         "mapping_id": str(mapping["_id"]),
#                         "zone_index": i,
#                         "coordinates": zone_data["roi"]
#                     })
#                     print(f"    Found ROI in zone {i}: {zone_data['roi']}")
    
#     print(f"Total ROI zones found: {len(roi_zones)}")
#     return roi_zones

# def draw_roi_zones(frame, roi_zones, frame_width, frame_height):
#     """
#     Draw ROI zones on the frame
    
#     Args:
#         frame: OpenCV frame
#         roi_zones: List of ROI zone data
#         frame_width: Frame width
#         frame_height: Frame height
#     """
#     for i, zone in enumerate(roi_zones):
#         coordinates = zone["coordinates"]
        
#         # Handle different coordinate formats
#         if isinstance(coordinates, list) and len(coordinates) >= 4:
#             # Assume format [x1, y1, x2, y2] or [(x1,y1), (x2,y2), ...]
#             if isinstance(coordinates[0], (list, tuple)) and len(coordinates[0]) == 2:
#                 # Format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
#                 points = [(int(x), int(y)) for x, y in coordinates[:4]]
                
#                 # Draw polygon
#                 pts = np.array(points, np.int32)
#                 pts = pts.reshape((-1, 1, 2))
#                 cv2.polylines(frame, [pts], True, (0, 0, 255), 3)  # Magenta color
                
#             else:
#                 # Format: [x1, y1, x2, y2]
#                 x1, y1, x2, y2 = map(int, coordinates[:4])
                
#                 # Ensure coordinates are within frame bounds
#                 x1 = max(0, min(x1, frame_width-1))
#                 y1 = max(0, min(y1, frame_height-1))
#                 x2 = max(0, min(x2, frame_width-1))
#                 y2 = max(0, min(y2, frame_height-1))
                
#                 # Draw rectangle
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)  # Magenta color
        
#         # Add zone label
#         zone_label = f"Zone {i+1}"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.4
#         thickness = 1
        
#         # Calculate text position (top-left of the zone)
#         if isinstance(coordinates[0], (list, tuple)):
#             text_x, text_y = int(coordinates[0][0]), int(coordinates[0][1])
#         else:
#             text_x, text_y = int(coordinates[0]), int(coordinates[1])
        
#         # Draw text background
#         (text_width, text_height), baseline = cv2.getTextSize(zone_label, font, font_scale, thickness)
#         cv2.rectangle(frame, 
#                      (text_x, text_y - text_height - baseline), 
#                      (text_x + text_width, text_y), 
#                      (255, 0, 255), cv2.FILLED)
        
#         # Draw text
#         cv2.putText(frame, zone_label, (text_x, text_y - baseline), 
#                    font, font_scale, (255, 255, 255), thickness)

# def process_metadata_for_duration(start_dt, end_dt, output_dir, mongo_uri, db_name, camera_name="Camera-018"):
#     client = MongoClient(mongo_uri)
#     db = client[db_name]
#     coll = db["metadata"]
    
#     # Get ROI zones for the camera
#     roi_zones = get_camera_roi_zones(client, db_name, camera_name)
    
#     if camera_name == None:
#         cursor = coll.find({
#             "time_stamp": {"$gte": start_dt, "$lte": end_dt}
#         })
#     else:
#         cursor = coll.find({
#             "time_stamp": {"$gte": start_dt, "$lte": end_dt},
#             "device_name": camera_name
#         })

#     # 1) Group metadata by video & frame, with robust frame number handling
#     videos = {}
#     for doc in cursor:
#         path = doc["evidence_path"]
        
#         # Try different frame number fields and convert safely
#         frame_num_raw = doc.get("evidence_frame_number", doc.get("frame_number", 0))
#         frame_num = safe_int_conversion(frame_num_raw)
        
#         # Store both the converted number and timestamp for sorting
#         frame_data = {
#             "frame_num": frame_num,
#             "timestamp": doc["time_stamp"],
#             "raw_frame_num": frame_num_raw
#         }
        
#         entries = []
#         for _, info in doc["track_ids_info"].items():
#             entries.append({
#                 "label_name": info["label_name"],
#                 "bbox": info["bbox"]
#             })
#             print(f"Found: {info['label_name']} at {info['bbox']} (frame: {frame_num_raw})")
        
#         if path not in videos:
#             videos[path] = {}
#         videos[path][frame_num] = {
#             "metadata": frame_data,
#             "annotations": entries
#         }

#     # 2) Process each video
#     for src_path, frames_data in videos.items():
#         if len(frames_data) < 20:
#             print(f"Skipping {src_path}: only {len(frames_data)} frames with metadata")
#             continue
            
#         if not os.path.exists(src_path):
#             print(f"[WARN] source not found: {src_path}; skipping.")
#             continue

#         print(f"\n=== PROCESSING {src_path} ===")
        
#         # Sort frames by timestamp to get proper order
#         frame_items = list(frames_data.items())
#         frame_items.sort(key=lambda x: x[1]["metadata"]["timestamp"])
        
#         print(f"Total frames with metadata: {len(frame_items)}")
#         print(f"Frame number range: {frame_items[0][1]['metadata']['raw_frame_num']} to {frame_items[-1][1]['metadata']['raw_frame_num']}")
        
#         # Create mapping from video frame index to annotations
#         frame_to_annotations = {}
#         for i, (_, frame_data) in enumerate(frame_items):
#             frame_to_annotations[i] = frame_data["annotations"]
        
#         print(f"Mapped to video frame indices: 0 to {len(frame_items)-1}")
#         print(f"ROI zones to be plotted: {len(roi_zones)}")

#         # --- Setup output directory ---
#         parent_dir = os.path.dirname(src_path)
#         rel_subdir = os.path.relpath(parent_dir, start="results/raw_videos")
#         output_parent = os.path.join(output_dir, rel_subdir)
#         os.makedirs(output_parent, exist_ok=True)

#         # --- Process video ---
#         cap = cv2.VideoCapture(src_path)
#         if not cap.isOpened():
#             print(f"[WARN] failed to open {src_path}; skipping.")
#             continue

#         fps = cap.get(cv2.CAP_PROP_FPS) or 25
#         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#         print(f"Video info: {w}x{h}, {fps} fps, {total_frames} total frames")

#         out_name = os.path.splitext(os.path.basename(src_path))[0] + "_annotated_with_zones.mp4"
#         out_fp = os.path.join(output_parent, out_name)
#         writer = cv2.VideoWriter(out_fp, fourcc, fps, (w, h))

#         print(f"Annotating {src_path} → {out_fp}")
        
#         frame_idx = 0
#         annotations_applied = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Draw ROI zones on every frame (persistent overlay)
#             if roi_zones:
#                 draw_roi_zones(frame, roi_zones, w, h)

#             # Apply detection annotations if this frame has them
#             if frame_idx in frame_to_annotations:
#                 print(f"  Annotating frame {frame_idx}")
                
#                 for obj in frame_to_annotations[frame_idx]:
#                     x1, y1, x2, y2 = obj["bbox"]
                    
#                     # Ensure coordinates are integers and within bounds
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     x1 = max(0, min(x1, w-1))
#                     y1 = max(0, min(y1, h-1))
#                     x2 = max(0, min(x2, w-1))
#                     y2 = max(0, min(y2, h-1))
                    
#                     # Skip invalid bboxes
#                     if x2 <= x1 or y2 <= y1:
#                         print(f"    [WARN] Invalid bbox: {obj['bbox']}")
#                         continue
                    
#                     # Draw detection bbox in green
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                     # Prepare label text
#                     text = obj["label_name"]
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     fs = 0.5
#                     th = 1

#                     # Measure text size
#                     (tw, tht), baseline = cv2.getTextSize(text, font, fs, th)

#                     # Draw text background
#                     rect_tl = (x1, max(0, y1 - tht - baseline - 2))
#                     rect_br = (min(w, x1 + tw), y1)
#                     cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)

#                     # Draw text
#                     text_org = (x1, max(tht, y1 - baseline - 1))
#                     cv2.putText(frame, text, text_org, font, fs, (255, 255, 255), th)
                
#                 annotations_applied += 1

#             writer.write(frame)
#             frame_idx += 1

#         cap.release()
#         writer.release()
        
#         print(f"  ✓ Applied annotations to {annotations_applied} frames")
#         print(f"  ✓ Applied ROI zones to all {frame_idx} frames")
#         print(f"  ✓ Processed {frame_idx} total frames")
        
#         if annotations_applied == 0:
#             print(f"  ⚠️  WARNING: No detection annotations applied!")
#             print(f"     Video frames: 0 to {frame_idx-1}")
#             print(f"     Available annotation frames: {sorted(frame_to_annotations.keys())}")
#         else:
#             print(f"  🎉 SUCCESS: Annotated {annotations_applied}/{len(frame_to_annotations)} available frames")

#     print("Done. Annotated videos with ROI zones in:", output_dir)


# if __name__ == "__main__":
#     # Add numpy import here since it's used in draw_roi_zones
#     import numpy as np
    
#     load_dotenv()

#     # parse defaults from .env or fallbacks
#     default_mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
#     parsed = urlparse(default_mongo_uri)
#     default_db = parsed.path.lstrip("/") or "mydatabase"

#     default_start_time = "2025-06-26T07:10:00.076+00:00"
#     default_end_time = "2025-06-26T07:10:59.821+00:00"
#     default_output_dir = "./results/plotted_videos/"

#     p = argparse.ArgumentParser(
#         description="Annotate videos with bounding boxes from metadata and ROI zones from camera mapping."
#     )
#     p.add_argument("--start-time", default=default_start_time,
#                    help=f"ISO start (default {default_start_time})")
#     p.add_argument("--end-time", default=default_end_time,
#                    help=f"ISO end   (default {default_end_time})")
#     p.add_argument("--output-dir", default=default_output_dir,
#                    help=f"Where to save annotated vids (default {default_output_dir})")
#     p.add_argument("--mongo-uri", default=default_mongo_uri,
#                    help="Mongo URI (from .env MONGODB_URI if set)")
#     p.add_argument("--db-name", default=default_db,
#                    help="MongoDB database name")
#     p.add_argument("--camera-name", default="Camera-018",
#                    help="Camera device name to process")
#     p.add_argument("--model-name", default="",
#                    help="(unused) model name placeholder")
#     args = p.parse_args()

#     # parse & force UTC
#     try:
#         st = datetime.datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
#         et = datetime.datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
#         start_utc = st if st.tzinfo else st.replace(tzinfo=datetime.timezone.utc)
#         end_utc = et if et.tzinfo else et.replace(tzinfo=datetime.timezone.utc)
#     except ValueError as ve:
#         print(f"Error: bad date format: {ve}. Use YYYY-MM-DDTHH:MM:SS")
#         exit(1)

#     print("Mongo URI:", args.mongo_uri)
#     print("DB name:  ", args.db_name)
#     print("Camera:   ", args.camera_name)
#     print("Time range:", start_utc, "→", end_utc)
#     print("Output dir:", args.output_dir)

#     # health‐check connect/disconnect
#     try:
#         me.connect(db=args.db_name, host=args.mongo_uri, uuidRepresentation="standard")
#         print("✔ Connected (mongoengine)")
#     except Exception as e:
#         print("✖ Connect error:", e)
#         exit(1)

#     process_metadata_for_duration(
#         start_dt=start_utc,
#         end_dt=end_utc,
#         output_dir=args.output_dir,
#         mongo_uri=args.mongo_uri,
#         db_name=args.db_name,
#         camera_name=args.camera_name
#     )

#     try:
#         me.disconnect()
#         print("✔ Disconnected (mongoengine)")
#     except Exception as e:
#         print("✖ Disconnect error:", e)

#     print("Script finished.")


"""
annotate_videos.py - ENHANCED VERSION WITH ROI ZONES - FIXED COORDINATE HANDLING

This version handles various frame numbering schemes and adds ROI zone plotting from camera_zone_mapping collection.
Fixed to handle complex nested coordinate structures.

Features:
- Plots bounding boxes from metadata
- Plots ROI zones from camera zone mapping with robust coordinate parsing
- Handles various frame numbering schemes
- Robust error handling

Usage example:
    python annotate_videos.py \
      --start-time "2025-06-13T12:00:00" \
      --end-time   "2025-06-13T12:10:00" \
      --output-dir "./results/plotted_videos" \
      --mongo-uri "mongodb://user:pass@host:27017/mydb" \
      --db-name   "mydb"
"""
import os
import argparse
import datetime
from urllib.parse import urlparse
from dateutil import parser as date_parser

from dotenv import load_dotenv
import mongoengine as me
from pymongo import MongoClient
from bson import ObjectId
import cv2
import numpy as np

def safe_int_conversion(value):
    """Safely convert various frame number formats to integers"""
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        try:
            # Try direct conversion first
            return int(value)
        except ValueError:
            try:
                # Try hexadecimal conversion
                return int(value, 16)
            except ValueError:
                # If all else fails, use hash of the string for consistent ordering
                return hash(value) % (2**31)
    else:
        # For other types (like ObjectId), convert to string then hash
        return hash(str(value)) % (2**31)

def flatten_coordinates(coords):
    """
    Recursively flatten nested coordinate structures to extract individual points
    
    Args:
        coords: Any nested structure containing coordinates
        
    Returns:
        list: Flattened list of coordinate pairs [(x1,y1), (x2,y2), ...]
    """
    def extract_points(data):
        points = []
        
        if isinstance(data, (list, tuple)):
            # Check if this is a coordinate pair [x, y] or (x, y)
            if len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
                points.append((float(data[0]), float(data[1])))
            else:
                # Recursively process each element
                for item in data:
                    points.extend(extract_points(item))
        elif isinstance(data, dict):
            # Check for common coordinate field names
            if 'x' in data and 'y' in data:
                points.append((float(data['x']), float(data['y'])))
            elif 'lat' in data and 'lon' in data:
                points.append((float(data['lon']), float(data['lat'])))  # Note: lon=x, lat=y
            else:
                # Recursively process dict values
                for value in data.values():
                    points.extend(extract_points(value))
        elif isinstance(data, (int, float)):
            # Single number - might be part of a flattened coordinate list
            return [data]
        
        return points
    
    points = extract_points(coords)
    
    # If we got a flat list of numbers, try to pair them up
    if len(points) > 0 and all(isinstance(p, (int, float)) for p in points):
        paired_points = []
        for i in range(0, len(points) - 1, 2):
            paired_points.append((points[i], points[i + 1]))
        return paired_points
    
    return points

def parse_roi_coordinates(coordinates):
    """
    Parse ROI coordinates from various formats and return a list of points
    
    Args:
        coordinates: Can be various formats:
            - [x1, y1, x2, y2] (rectangle)
            - [[x1, y1], [x2, y2], ...] (polygon points)
            - [[[x1, y1]], [[x2, y2]], ...] (nested structure)
            - Complex nested structures
    
    Returns:
        tuple: (points_list, shape_type)
            - points_list: [(x1,y1), (x2,y2), ...] 
            - shape_type: 'rectangle' or 'polygon'
    """
    print(f"    Parsing coordinates: {coordinates}")
    print(f"    Coordinate type: {type(coordinates)}")
    
    if not coordinates:
        return [], 'polygon'
    
    # Flatten the coordinates
    points = flatten_coordinates(coordinates)
    print(f"    Flattened to points: {points}")
    
    if len(points) < 2:
        print(f"    [WARN] Not enough points: {len(points)}")
        return [], 'polygon'
    
    # Convert to integers and ensure they're valid
    valid_points = []
    for point in points:
        try:
            x, y = int(float(point[0])), int(float(point[1]))
            # Basic sanity check - coordinates should be reasonable
            if -10000 <= x <= 10000 and -10000 <= y <= 10000:
                valid_points.append((x, y))
        except (ValueError, TypeError, IndexError):
            print(f"    [WARN] Invalid point: {point}")
            continue
    
    if len(valid_points) < 2:
        print(f"    [WARN] No valid points after filtering")
        return [], 'polygon'
    
    # Determine shape type
    if len(valid_points) == 2:
        # Two points - treat as rectangle diagonal
        shape_type = 'rectangle'
    elif len(valid_points) == 4:
        # Four points - could be rectangle corners or polygon
        # Check if it looks like rectangle format [x1,y1,x2,y2] converted to points
        p1, p2, p3, p4 = valid_points
        if (p1[0] == p3[0] and p2[0] == p4[0] and  # vertical alignment
            p1[1] == p2[1] and p3[1] == p4[1]):    # horizontal alignment
            shape_type = 'rectangle'
        else:
            shape_type = 'polygon'
    else:
        shape_type = 'polygon'
    
    print(f"    Result: {len(valid_points)} points, shape_type: {shape_type}")
    return valid_points, shape_type

def get_camera_roi_zones(client, db_name, camera_name="Camera-018"):
    """
    Get ROI zones for a camera by matching cameras collection with camera_zone_mapping
    ROI coordinates are directly in the camera_zone_mapping collection
    
    Returns:
        list: List of ROI zones with coordinates
    """
    db = client[db_name]
    cameras_coll = db["cameras"]
    zone_mapping_coll = db["camera_zone_mapping"]
    
    print(f"\n=== FETCHING ROI ZONES FOR {camera_name} ===")
    
    # Find camera document by device_name
    camera_doc = cameras_coll.find_one({"device_name": camera_name})
    if not camera_doc:
        print(f"[WARN] Camera {camera_name} not found in cameras collection")
        return []
    
    camera_object_id = camera_doc["_id"]
    print(f"Found camera with ObjectId: {camera_object_id}")
    
    # Find zone mappings for this camera
    zone_mappings = zone_mapping_coll.find({"camera": camera_object_id})
    
    roi_zones = []
    for mapping in zone_mappings:
        print(f"Found zone mapping: {mapping['_id']}")
        
        # Extract ROI directly from the mapping document
        if "roi" in mapping:
            roi_coordinates = mapping["roi"]
            print(f"  Found ROI coordinates: {roi_coordinates}")
            
            # Parse the coordinates
            points, shape_type = parse_roi_coordinates(roi_coordinates)
            if points:
                roi_zones.append({
                    "mapping_id": str(mapping["_id"]),
                    "coordinates": points,
                    "shape_type": shape_type,
                    "label": f"ROI-{len(roi_zones)+1}"
                })
                print(f"  ✓ Added ROI zone with {len(points)} points ({shape_type})")
            else:
                print(f"  ✗ Failed to parse ROI coordinates")
        else:
            print(f"  [WARN] No ROI found in mapping {mapping['_id']}")
        
        # Also check if there are zones array with individual ROIs
        if "zones" in mapping and isinstance(mapping["zones"], list):
            for i, zone_data in enumerate(mapping["zones"]):
                print(f"  Checking zone {i}: {zone_data}")
                
                # If zone_data is a dict with ROI
                if isinstance(zone_data, dict) and "roi" in zone_data:
                    points, shape_type = parse_roi_coordinates(zone_data["roi"])
                    if points:
                        roi_zones.append({
                            "mapping_id": str(mapping["_id"]),
                            "zone_index": i,
                            "coordinates": points,
                            "shape_type": shape_type,
                            "label": f"Zone-{i+1}"
                        })
                        print(f"    ✓ Added zone {i} with {len(points)} points ({shape_type})")
                    else:
                        print(f"    ✗ Failed to parse zone {i} coordinates")
    
    print(f"Total ROI zones found: {len(roi_zones)}")
    return roi_zones

def draw_roi_zones(frame, roi_zones, frame_width, frame_height):
    """
    Draw ROI zones on the frame
    
    Args:
        frame: OpenCV frame
        roi_zones: List of ROI zone data
        frame_width: Frame width
        frame_height: Frame height
    """
    for i, zone in enumerate(roi_zones):
        points = zone["coordinates"]
        shape_type = zone.get("shape_type", "polygon")
        label = zone.get("label", f"Zone {i+1}")
        
        if not points:
            continue
        
        # Ensure all points are within frame bounds
        clipped_points = []
        for x, y in points:
            x_clipped = max(0, min(x, frame_width - 1))
            y_clipped = max(0, min(y, frame_height - 1))
            clipped_points.append((x_clipped, y_clipped))
        
        if len(clipped_points) < 2:
            continue
        
        # Draw based on shape type
        if shape_type == 'rectangle' and len(clipped_points) >= 2:
            # Draw rectangle - use first two points as opposite corners
            p1, p2 = clipped_points[0], clipped_points[1]
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red color
            text_x, text_y = x1, y1
            
        else:
            # Draw polygon
            if len(clipped_points) >= 3:
                pts = np.array(clipped_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 0, 255), 3)  # Red color
                text_x, text_y = clipped_points[0]
            elif len(clipped_points) == 2:
                # Fallback to line if only 2 points
                p1, p2 = clipped_points
                cv2.line(frame, p1, p2, (0, 0, 255), 3)
                text_x, text_y = p1
            else:
                continue
        
        # Add zone label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position text above the zone
        bg_y1 = max(0, text_y - text_height - baseline - 4)
        bg_y2 = text_y
        bg_x1 = text_x
        bg_x2 = min(frame_width, text_x + text_width + 4)
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), cv2.FILLED)
        
        # Draw text
        text_org = (text_x + 2, text_y - baseline - 2)
        cv2.putText(frame, label, text_org, font, font_scale, (255, 255, 255), thickness)
        
        print(f"    Drew {shape_type} zone '{label}' with {len(clipped_points)} points")

def process_metadata_for_duration(start_dt, end_dt, output_dir, mongo_uri, db_name, camera_name="Camera-018"):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db["metadata"]
    
    # Get ROI zones for the camera
    roi_zones = get_camera_roi_zones(client, db_name, camera_name)
    
    if camera_name == None:
        cursor = coll.find({
            "time_stamp": {"$gte": start_dt, "$lte": end_dt}
        })
    else:
        cursor = coll.find({
            "time_stamp": {"$gte": start_dt, "$lte": end_dt},
            "device_name": camera_name
        })

    # 1) Group metadata by video & frame, with robust frame number handling
    videos = {}
    for doc in cursor:
        path = doc["evidence_path"]
        
        # Try different frame number fields and convert safely
        frame_num_raw = doc.get("evidence_frame_number", doc.get("frame_number", 0))
        frame_num = safe_int_conversion(frame_num_raw)
        
        # Store both the converted number and timestamp for sorting
        frame_data = {
            "frame_num": frame_num,
            "timestamp": doc["time_stamp"],
            "raw_frame_num": frame_num_raw
        }
        
        entries = []
        for _, info in doc["track_ids_info"].items():
            entries.append({
                "label_name": info["label_name"],
                "bbox": info["bbox"],
                # "track_id": info["track_id"]
            })
            print(f"Found: {info['label_name']} at {info['bbox']} (frame: {frame_num_raw})")
        
        if path not in videos:
            videos[path] = {}
        videos[path][frame_num] = {
            "metadata": frame_data,
            "annotations": entries
        }

    # print("videos", videos)

    # 2) Process each video
    for src_path, frames_data in videos.items():
        if len(frames_data) < 20:
            print(f"Skipping {src_path}: only {len(frames_data)} frames with metadata")
            continue
            
        if not os.path.exists(src_path):
            print(f"[WARN] source not found: {src_path}; skipping.")
            continue

        print(f"\n=== PROCESSING {src_path} ===")
        
        # Sort frames by timestamp to get proper order
        frame_items = list(frames_data.items())
        frame_items.sort(key=lambda x: x[1]["metadata"]["timestamp"])
        
        print(f"Total frames with metadata: {len(frame_items)}")
        print(f"Frame number range: {frame_items[0][1]['metadata']['raw_frame_num']} to {frame_items[-1][1]['metadata']['raw_frame_num']}")
        
        # Create mapping from video frame index to annotations
        frame_to_annotations = {}
        for i, (_, frame_data) in enumerate(frame_items):
            frame_to_annotations[i] = frame_data["annotations"]
        
        print(f"Mapped to video frame indices: 0 to {len(frame_items)-1}")
        print(f"ROI zones to be plotted: {len(roi_zones)}")

        # --- Setup output directory ---
        parent_dir = os.path.dirname(src_path)
        rel_subdir = os.path.relpath(parent_dir, start="results/raw_videos")
        output_parent = os.path.join(output_dir, rel_subdir)
        os.makedirs(output_parent, exist_ok=True)
        print("output_parent:", output_parent)

        # --- Process video ---
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            print(f"[WARN] failed to open {src_path}; skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        print(f"Video info: {w}x{h}, {fps} fps, {total_frames} total frames")

        out_name = os.path.splitext(os.path.basename(src_path))[0] + "_annotated_with_zones.mp4"
        out_fp = os.path.join(output_parent, out_name)
        writer = cv2.VideoWriter(out_fp, fourcc, fps, (w, h))

        print(f"Annotating {src_path} → {out_fp}")
        
        frame_idx = 0
        annotations_applied = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw ROI zones on every frame (persistent overlay)
            if roi_zones:
                draw_roi_zones(frame, roi_zones, w, h)

            # Apply detection annotations if this frame has them
            if frame_idx in frame_to_annotations:
                print(f"  Annotating frame {frame_idx}")
                
                for obj in frame_to_annotations[frame_idx]:
                    x1, y1, x2, y2 = obj["bbox"]
                    
                    # Ensure coordinates are integers and within bounds
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    # Skip invalid bboxes
                    if x2 <= x1 or y2 <= y1:
                        print(f"    [WARN] Invalid bbox: {obj['bbox']}")
                        continue
                    
                    # Draw detection bbox in green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Prepare label text
                    text = obj["label_name"]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fs = 0.5
                    th = 1

                    # Measure text size
                    (tw, tht), baseline = cv2.getTextSize(text, font, fs, th)

                    # Draw text background
                    rect_tl = (x1, max(0, y1 - tht - baseline - 2))
                    rect_br = (min(w, x1 + tw), y1)
                    cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)

                    # Draw text
                    text_org = (x1, max(tht, y1 - baseline - 1))
                    cv2.putText(frame, text, text_org, font, fs, (255, 255, 255), th)
                
                annotations_applied += 1

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        
        print(f"  ✓ Applied annotations to {annotations_applied} frames")
        print(f"  ✓ Applied ROI zones to all {frame_idx} frames")
        print(f"  ✓ Processed {frame_idx} total frames")
        
        if annotations_applied == 0:
            print(f"  ⚠️  WARNING: No detection annotations applied!")
            print(f"     Video frames: 0 to {frame_idx-1}")
            print(f"     Available annotation frames: {sorted(frame_to_annotations.keys())}")
        else:
            print(f"  🎉 SUCCESS: Annotated {annotations_applied}/{len(frame_to_annotations)} available frames")

    print("Done. Annotated videos with ROI zones in:", output_dir)


if __name__ == "__main__":
    load_dotenv()

    # parse defaults from .env or fallbacks
    default_mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    parsed = urlparse(default_mongo_uri)
    default_db = parsed.path.lstrip("/") or "mydatabase"

    default_start_time = "2025-06-29T13:00:00.076+00:00"
    default_end_time = "2025-06-29T13:06:00.821+00:00"
    default_output_dir = "./results/plotted_videos/"

    p = argparse.ArgumentParser(
        description="Annotate videos with bounding boxes from metadata and ROI zones from camera mapping."
    )
    p.add_argument("--start-time", default=default_start_time,
                   help=f"ISO start (default {default_start_time})")
    p.add_argument("--end-time", default=default_end_time,
                   help=f"ISO end   (default {default_end_time})")
    p.add_argument("--output-dir", default=default_output_dir,
                   help=f"Where to save annotated vids (default {default_output_dir})")
    p.add_argument("--mongo-uri", default=default_mongo_uri,
                   help="Mongo URI (from .env MONGODB_URI if set)")
    p.add_argument("--db-name", default=default_db,
                   help="MongoDB database name")
    p.add_argument("--camera-name", default="Camera-010",
                   help="Camera device name to process")
    p.add_argument("--model-name", default="",
                   help="(unused) model name placeholder")
    args = p.parse_args()

    # parse & force UTC
    try:
        st = datetime.datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
        et = datetime.datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
        start_utc = st if st.tzinfo else st.replace(tzinfo=datetime.timezone.utc)
        end_utc = et if et.tzinfo else et.replace(tzinfo=datetime.timezone.utc)
    except ValueError as ve:
        print(f"Error: bad date format: {ve}. Use YYYY-MM-DDTHH:MM:SS")
        exit(1)

    print("Mongo URI:", args.mongo_uri)
    print("DB name:  ", args.db_name)
    print("Camera:   ", args.camera_name)
    print("Time range:", start_utc, "→", end_utc)
    print("Output dir:", args.output_dir)

    # health‐check connect/disconnect
    try:
        me.connect(db=args.db_name, host=args.mongo_uri, uuidRepresentation="standard")
        print("✔ Connected (mongoengine)")
    except Exception as e:
        print("✖ Connect error:", e)
        exit(1)

    process_metadata_for_duration(
        start_dt=start_utc,
        end_dt=end_utc,
        output_dir=args.output_dir,
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        camera_name=args.camera_name
    )

    try:
        me.disconnect()
        print("✔ Disconnected (mongoengine)")
    except Exception as e:
        print("✖ Disconnect error:", e)

    print("Script finished.")