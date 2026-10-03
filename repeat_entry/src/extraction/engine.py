import os
import sys
import pickle
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np

# Handle pathing for submodules
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Package imports
import config as local_config
from src.model import TripletDino


class FootfallBatchDataset(Dataset):
    """
    Dataset to handle batch loading of footfall images across all tracks.
    """
    def __init__(self, items):
        """
        Args:
            items (list): List of (image_path, track_key) tuples.
        """
        self.items = items
        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=Image.Resampling.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, track_key = self.items[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(image)
            return img_tensor, img_path, track_key
        except Exception as e:
            # Return zero tensor to maintain batch alignment
            return torch.zeros(3, 224, 224), img_path, track_key

# def load_model_parallel():
#     """
#     Loads TripletDino and wraps it in DataParallel for Multi-GPU utilization.
#     """
#     model_path = local_config.MODEL_PATH
    
#     # Base Model Selection
#     if getattr(local_config, 'USE_BASE_MODEL', False):
#         print(f"🔹 Using Multi-GPU BASE DINOv3 ({local_config.DINOV3_MODEL_NAME})")
#         model = TripletDino(embedding_dim=256, model_path=None)
#     elif model_path and os.path.exists(model_path):
#         print(f"🔹 Loading Multi-GPU Fine-tuned model from {model_path}")
#         model = TripletDino(embedding_dim=256, model_path=model_path)
#         checkpoint = torch.load(model_path, map_location='cpu')
#         state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
#         model.load_state_dict(state_dict)
#     else:
#         print(f"⚠️ Custom model not found. Using BASE DINOv3.")
#         model = TripletDino(embedding_dim=256, model_path=None)

#     # Multi-GPU Initialization
#     if torch.cuda.device_count() > 1:
#         print(f"🚀 Engaging Multi-GPU Engine: {torch.cuda.device_count()} GPUs detected.")
#         model = torch.nn.DataParallel(model)
    
#     model.to(local_config.DEVICE)
#     model.eval()
#     return model

from src.backbone import DinoV2Backbone


class ReIDWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        emb, _ = self.model(x)

        return emb


def load_model_parallel():
    """
    Loads your trained DinoV2 ReID model
    and wraps it for extraction pipeline.
    """

    model_path = local_config.MODEL_PATH

    print(
        f"🔹 Loading ReID model from: "
        f"{model_path}"
    )

    # --------------------------------
    # LOAD MODEL
    # --------------------------------

    base_model = DinoV2Backbone(
        num_classes=174,     # <-- change if needed
        unfreeze_n=2
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    base_model.load_state_dict(
        state_dict
    )

    # --------------------------------
    # WRAP
    # extraction engine expects:
    #
    # embs = model(imgs)
    #
    # while your model returns:
    #
    # emb, logits
    # --------------------------------

    model = ReIDWrapper(
        base_model
    )

    # --------------------------------
    # MULTI GPU
    # --------------------------------

    if False:

        print(
            f"🚀 Engaging Multi-GPU Engine: "
            f"{torch.cuda.device_count()} GPUs detected."
        )

        model = torch.nn.DataParallel(
            model
        )

    model.to(
        local_config.DEVICE
    )

    model.eval()

    return model

def run_extraction(logger, output_dir=None):
    """
    High-performance batched extraction engine.
    """
    logger.info("--- Stage: High-Throughput Batched Extraction ---")
    
    # 1. Prepare Master Cache
    all_embeddings = {}
    if os.path.exists(local_config.EMBEDDINGS_PKL) and not local_config.FORCE_REEXTRACT:
        with open(local_config.EMBEDDINGS_PKL, 'rb') as f:
            all_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(all_embeddings)} existing tracks from master cache.")
    
    # 2. Harvest all images needing processing
    harvested_items = []
    days = sorted([
        d for d in os.listdir(local_config.FOOTFALL_ROOT)
        if os.path.isdir(os.path.join(local_config.FOOTFALL_ROOT, d))
        and d == local_config.TODAY_STR  # Only process the configured target date
    ])
    
    for day in days:
        day_path = os.path.join(local_config.FOOTFALL_ROOT, day)
        tracks = sorted([t for t in os.listdir(day_path) if os.path.isdir(os.path.join(day_path, t))])
        
        for track_id in tracks:
            full_track_key = f"{day}/{track_id}"
            if full_track_key in all_embeddings and not local_config.FORCE_REEXTRACT:
                continue
            
            track_path = os.path.join(day_path, track_id)
            img_paths = sorted([os.path.join(track_path, f) for f in os.listdir(track_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            # --- Temporal Sampling Logic (Equidistant + Noise Buffers) ---
            if len(img_paths) > 0:
                from src.utils.visualizer import parse_footfall_timestamp
                
                # 1. Gather timestamps
                path_ts = []
                for p in img_paths:
                    ts = parse_footfall_timestamp(p)
                    if ts: path_ts.append((p, ts))
                
                if path_ts:
                    # Ensure sorted by time
                    path_ts.sort(key=lambda x: x[1])
                    
                    # 2. Apply Noise Buffers
                    start_t = path_ts[0][1]
                    end_t = path_ts[-1][1]
                    tr = local_config.TRIM_BUFFER_SECONDS
                    
                    stable_pool = [p for p, ts in path_ts if (ts - start_t).total_seconds() >= tr and (end_t - ts).total_seconds() >= tr]
                    
                    # Fallback to full pool if trimming is too aggressive
                    candidate_pool = stable_pool if len(stable_pool) >= local_config.MAX_SAMPLES_PER_TRACK else [p for p, ts in path_ts]
                    
                    # 3. Equidistant Sampling
                    n = len(candidate_pool)
                    k = local_config.MAX_SAMPLES_PER_TRACK
                    if n > k:
                        indices = np.linspace(0, n - 1, k).astype(int)
                        img_paths = [candidate_pool[i] for i in indices]
                    else:
                        img_paths = candidate_pool

            for p in img_paths:
                harvested_items.append((p, full_track_key))
                
    if not harvested_items:
        logger.info("No new tracks detected. Extraction complete.")
    else:
        logger.info(f"📊 Scheduled {len(harvested_items)} images for batched extraction.")
        
        # 3. Initialise Batched Pipeline
        model = load_model_parallel()
        dataset = FootfallBatchDataset(harvested_items)
        dataloader = DataLoader(
            dataset, 
            batch_size=local_config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=local_config.NUM_WORKERS,
            pin_memory=True
        )
        
        # 4. Global Inference Loop
        new_results = {}
        for imgs, paths, keys in tqdm(dataloader, desc="GPUBatch"):
            imgs = imgs.to(local_config.DEVICE)
            with torch.no_grad():
                embs = model(imgs).cpu()
            
            # Re-collect into tracks
            for i in range(len(paths)):
                k = keys[i]
                p = paths[i]
                e = embs[i]
                if k not in new_results:
                    new_results[k] = []
                new_results[k].append((p, e))
        
        # 5. Merge and Save
        all_embeddings.update(new_results)
        logger.info(f"Extracted features for {len(new_results)} new tracks.")
        
        Path(local_config.EMBEDDINGS_PKL).parent.mkdir(parents=True, exist_ok=True)
        with open(local_config.EMBEDDINGS_PKL, 'wb') as f:
            pickle.dump(all_embeddings, f)
        logger.info(f"Master embeddings cache updated: {local_config.EMBEDDINGS_PKL}")

    # 6. Secure Snapshot Backup
    if output_dir and os.path.exists(local_config.EMBEDDINGS_PKL):
        backup_path = os.path.join(output_dir, os.path.basename(local_config.EMBEDDINGS_PKL))
        shutil.copy2(local_config.EMBEDDINGS_PKL, backup_path)
        logger.info(f"💾 Multi-GPU snapshot archived in run directory: {backup_path}")

if __name__ == "__main__":
    from src.utils.helpers import setup_logging
    logger = setup_logging(local_config.BASE_RESULTS_DIR, "standalone_gpu_batch")
    run_extraction(logger)
