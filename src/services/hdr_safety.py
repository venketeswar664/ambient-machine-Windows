#!/usr/bin/env python3
## src/services/hdr_safety.py
import argparse
import os
import sys
import time
import cv2
from bson import ObjectId
from pymongo import MongoClient
from ultralytics import YOLO
from shapely.geometry import Polygon, Point


HUMAN_LABEL = "humans" 

ZONES = {
    'HDR_23_24': Polygon([(828.61, 322.17), (828.61, 640.88), (1336.82, 635.13), (1342.56, 314.99)])
}

def is_person_in_zone(bbox, zone_name='HDR_23_24'):
    """
    Check if a person's bottom-center point is inside the specified zone.
    """
    if zone_name not in ZONES:
        print(f"[hdr_safety] Warning: Zone '{zone_name}' not found. Returning False.", flush=True)
        return False
    
    try:
        x1, y1, x2, y2 = map(int, bbox)
        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2  # Bottom edge
        bottom_center_point = Point(bottom_center_x, bottom_center_y)
        return ZONES[zone_name].contains(bottom_center_point)
    except Exception as e:
        print(f"[hdr_safety] Error in zone check: {e}", flush=True)
        return False


def load_model(model_path: str, device: str = "cuda:0", conf: float = 0.6):
    """
    Load the YOLO safety model.
    """
    try:
        model = YOLO(model_path)
        print(f"[hdr_safety] Safety model loaded: {model_path}", flush=True)
        return model, device, conf
    except Exception as e:
        print(f"[hdr_safety] ERROR loading model at {model_path}: {e}", flush=True)
        return None, device, conf

def predict_safety(model, device, conf, crop_img, bbox):
    """
    Run safety prediction on a cropped image and apply zone logic.
    """
    if model is None:
        return None
    try:
        results = model.predict(crop_img, verbose=False, conf=conf, device=device)
        names = results[0].names
        
        # Stores detection status
        pred = {"helmet": False, "no_helmet": False, "ppe": False, "no_ppe": False, "shoe": False, "no_shoe": False}
        
        for box in results[0].boxes:
            cls = int(box.cls)
            name = names.get(cls, str(cls))
            if name in pred:
                pred[name] = True

        # --- Apply hdr_safety logic ---
        
        # 1. Calculate detected status
        detected_helmet = pred["helmet"] and not pred["no_helmet"]
        detected_ppe = pred["ppe"] and not pred["no_ppe"]
        detected_shoes = pred["shoe"] and not pred["no_shoe"] 

        # 2. Check if person is in the required zone
        in_required_zone = is_person_in_zone(bbox, 'HDR_23_24')

        # 3. Determine final status based on zone logic
        # If NOT in the zone, helmet check is passed (True)
        # If IN the zone, helmet status depends on detection
        final_helmet = True if not in_required_zone else detected_helmet
        
        # PPE and shoes are required regardless of zone
        final_ppe = detected_ppe
        final_shoes = detected_shoes

        return {"helmet": final_helmet, "ppe": final_ppe, "shoes": final_shoes}
        # --- End of hdr_safety logic ---

    except Exception as e:
        print(f"[hdr_safety] inference error: {e}", flush=True)
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.6)
    args = ap.parse_args()

    MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://sentinel:sentinelMongo_test123@localhost:27018/sentinel_warehouse")
    MONGO_DB  = os.environ.get("MONGODB_DBNAME", "sentinel_warehouse")

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    print(f"[hdr_safety] using model_path={args.model_path}", flush=True)
    model, device, conf = load_model(args.model_path, args.device, args.conf)

    camera_oid = ObjectId(args.camera_id)
    project_root = os.getcwd()

    pipeline = [{'$match': {'operationType': 'insert', 'fullDocument.device': camera_oid}}]
    print(f"[hdr_safety] listening for metadata inserts for device={camera_oid}", flush=True)

    while True:
        try:
            with db["metadata"].watch(pipeline, full_document="updateLookup") as stream:
                for change in stream:
                    doc = change["fullDocument"]
                    raw_frame_path = doc.get("raw_frame_path")
                    if not raw_frame_path:
                        continue

                    ti = doc.get("track_ids_info", {})
                    if not any(v.get("label_name") == HUMAN_LABEL for v in ti.values()):
                        continue

                    abs_path = os.path.join(project_root, raw_frame_path)
                    if not os.path.exists(abs_path):
                        print(f"[hdr_safety] missing frame: {abs_path}", flush=True)
                        continue

                    frame = cv2.imread(abs_path)
                    if frame is None:
                        print(f"[hdr_safety] failed to read frame: {abs_path}", flush=True)
                        continue

                    updates = {}
                    for tid, info in ti.items():
                        if info.get("label_name") != HUMAN_LABEL:
                            continue
                        
                        bbox = info.get("bbox")
                        if not bbox or len(bbox) != 4:
                            continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size <= 0:
                            continue
                        
                        # Pass the original bbox for the zone check
                        res = predict_safety(model, device, conf, crop, bbox)
                        
                        if res is not None:
                            updates[f"track_ids_info.{tid}.safety_status"] = res

                    if updates:
                        db["metadata"].update_one({"_id": doc["_id"]}, {"$set": updates})

        except Exception as e:
            print(f"[hdr_safety] change stream error: {e} (retry in 5s)", flush=True)
            time.sleep(5)

if __name__ == "__main__":
    main()