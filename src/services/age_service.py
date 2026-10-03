import argparse
import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from bson import ObjectId
from pymongo import MongoClient
import numpy as np

CUSTOMER_LABEL = "class_0"
# Updated labels for Age classification
CLASS_NAMES = ["adult", "kid", "old"]

class DinoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Load DINOv2 backbone (Exact same backbone as gender)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14"
        )
        embed_dim = self.backbone.embed_dim

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 2 transformer blocks for fine-tuning
        for name, param in self.backbone.named_parameters():
            if "blocks.10" in name or "blocks.11" in name:
                param.requires_grad = True

        self.embedding_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)   
        )

        # Classifier head for 3 classes
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        embeddings = self.embedding_head(cls_token)
        embeddings = F.normalize(embeddings, dim=1)
        logits = self.classifier(embeddings)
        return embeddings, logits


class DinoInference:
    def __init__(self, model_path, class_names, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        # Initialize with 3 classes
        self.model = DinoClassifier(num_classes=len(class_names))
        
        try:
            # Note: Ensure the .pth file loaded here was trained for 3 classes
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[age_service] Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"[age_service] ERROR loading model weights: {e}", flush=True)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        return self.class_names[pred.item()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", required=True, help="Camera ObjectId")
    ap.add_argument("--model-path", required=True, help="Path to age model .pth file")
    ap.add_argument("--device", default="cpu", help="Device to run inference on")
    ap.add_argument("--date", default=None, help="Process historical data for a specific date (YYYY-MM-DD).")
    args = ap.parse_args()

    # Configuration (Consider moving these to a config file/env vars)
    MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://seawoods:sentinelMongo%40123@localhost:27017/sentinel?authSource=sentinel&replicaSet=rs0&directConnection=true")
    MONGO_DB = os.environ.get("MONGODB_DBNAME", "sentinel")

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    print(f"[age_service] Using model_path={args.model_path}", flush=True)
    inference = DinoInference(args.model_path, CLASS_NAMES, device=args.device)

    camera_oid = ObjectId(args.camera_id)
    project_root = os.getcwd()

    if args.date:
        import datetime
        print(f"[age_service] Running historical batch processing for date {args.date} and device {camera_oid}", flush=True)
        start_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        end_date = start_date + datetime.timedelta(days=1)
        
        query = {
            "device": camera_oid,
            "time_stamp": {"$gte": start_date, "$lt": end_date},
            "track_ids_info": {"$exists": True, "$ne": {}}
        }
        
        cursor = db["metadata"].find(query)
        docs_processed = 0
        for doc in cursor:
            raw_frame_path = doc.get("raw_frame_path")
            if not raw_frame_path: continue

            ti = doc.get("track_ids_info", {})
            if not any(v.get("label_name") == CUSTOMER_LABEL for v in ti.values()):
                continue

            abs_path = os.path.join(project_root, raw_frame_path.lstrip("./"))
            frame = cv2.imread(abs_path)
            if frame is None: continue

            updates = {}
            for tid, info in ti.items():
                if info.get("label_name") != CUSTOMER_LABEL:
                    continue
                
                # If already predicted, optionally skip? Or overwrite. User said "some cases still need to produces".
                if "age_prediction" in info:
                    continue
                
                bbox = info.get("bbox")
                if not bbox or len(bbox) != 4: continue
                
                x1, y1, x2, y2 = map(int, bbox)
                if x1 >= x2 or y1 >= y2: continue
                
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size <= 0: continue
                
                age_result = inference.predict(crop)
                if age_result:
                    updates[f"track_ids_info.{tid}.age_prediction"] = age_result
                    print(f"[age_service] Track {tid}: {age_result}", flush=True)

            if updates:
                db["metadata"].update_one({"_id": doc["_id"]}, {"$set": updates})
                docs_processed += 1
                
        print(f"[age_service] Finished processing date {args.date}. Updated {docs_processed} documents.", flush=True)
        return

    pipeline = [{'$match': {'operationType': 'insert', 'fullDocument.device': camera_oid}}]
    print(f"[age_service] Listening for metadata for device={camera_oid}", flush=True)

    while True:
        try:
            with db["metadata"].watch(pipeline, full_document="updateLookup") as stream:
                for change in stream:
                    doc = change["fullDocument"]
                    raw_frame_path = doc.get("raw_frame_path")
                    if not raw_frame_path: continue

                    ti = doc.get("track_ids_info", {})
                    if not any(v.get("label_name") == CUSTOMER_LABEL for v in ti.values()):
                        continue

                    abs_path = os.path.join(project_root, raw_frame_path.lstrip("./"))
                    frame = cv2.imread(abs_path)
                    if frame is None: continue

                    updates = {}
                    for tid, info in ti.items():
                        if info.get("label_name") != CUSTOMER_LABEL:
                            continue
                        
                        bbox = info.get("bbox")
                        if not bbox or len(bbox) != 4: continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size <= 0: continue
                        
                        age_result = inference.predict(crop)
                        if age_result:
                            updates[f"track_ids_info.{tid}.age_prediction"] = age_result
                            print(f"[age_service] Track {tid}: {age_result}", flush=True)

                    if updates:
                        db["metadata"].update_one({"_id": doc["_id"]}, {"$set": updates})
                        print(f"[age_service] Updated {len(updates)} customer(s)", flush=True)

        except Exception as e:
            print(f"[age_service] Stream error: {e}. Retrying...", flush=True)
            time.sleep(5)

if __name__ == "__main__":
    main()