import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F # <-- Import functional for normalization
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import config
from transformers import AutoImageProcessor, AutoModel
import torch.hub
import timm

# Vendored facebook/dinov2-base (repeat_entry/models/dinov2-base) — avoids any
# dependency on the HuggingFace hub cache location / network.
_LOCAL_DINOV2 = Path(__file__).resolve().parent.parent / "models" / "dinov2-base"
DINOV2_SOURCE = (
    str(_LOCAL_DINOV2)
    if (_LOCAL_DINOV2 / "config.json").is_file()
    else "facebook/dinov2-base"
)

# --- Dataset classes remain unchanged ---
class InferenceDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image), str(image_path)
        except Exception as e:
            logging.warning(f"Could not load image {image_path}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

class TripletDataset(Dataset):
    def __init__(self, data_map: Dict[str, List[Path]], processor):
        self.data_map = data_map
        self.processor = processor
        self.track_ids = list(data_map.keys())
        
        # --- UDA SPECIFIC MINING ---
        # Separate track IDs into formal (cohort) and current day
        self.cohort_track_ids = [tid for tid in self.track_ids if tid.startswith('c_')]
        self.current_day_track_ids = [tid for tid in self.track_ids if not tid.startswith('c_')]
        
        # The true anchor pool is STRICTLY the cohort tracks (to enforce cross-camera training)
        # If there's no cohort data, fallback safely
        anchor_tids_to_use = self.cohort_track_ids if self.cohort_track_ids else self.track_ids
        self.anchor_pool = [(p, tid) for tid in anchor_tids_to_use for p in data_map[tid]]
        
        if len(self.track_ids) < 2: raise ValueError(f"Need at least 2 unique track IDs for triplets, found {len(self.track_ids)}.")
        dino_input_size = self.processor.size.get('shortest_edge', 224)
        self.pil_transforms = T.Compose([
            T.Resize((dino_input_size, dino_input_size)), # Fixed resize to avoid aspect ratio distortion
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1), # More aggressive color focus
        ])
        # Slightly more aggressive erasing to prevent relying on logos/small specific features
        self.tensor_transforms = T.Compose([T.RandomErasing(p=0.7, scale=(0.05, 0.3), ratio=(0.3, 3.3), value=0)])
        
    def __len__(self): 
        return len(self.anchor_pool)
        
    def __getitem__(self, index):
        anchor_path, anchor_track_id = self.anchor_pool[index]
        
        # Positive comes entirely from the identical track (which spans cameras in cohort data!)
        positive_candidates = [p for p in self.data_map[anchor_track_id] if p != anchor_path]
        positive_path = random.choice(positive_candidates) if positive_candidates else anchor_path
        
        # Negative is drawn globally from BOTH cohort AND current day (Injects daily people as hard negs)
        negative_track_ids = [tid for tid in self.track_ids if tid != anchor_track_id]
        negative_track_id = random.choice(negative_track_ids) if negative_track_ids else anchor_track_id
        negative_path = random.choice(self.data_map.get(negative_track_id, []))
        
        anchor_img = self.pil_transforms(Image.open(anchor_path).convert("RGB"))
        positive_img = self.pil_transforms(Image.open(positive_path).convert("RGB"))
        negative_img = self.pil_transforms(Image.open(negative_path).convert("RGB"))
        
        anchor_inputs = self.tensor_transforms(self.processor(images=anchor_img, return_tensors="pt")['pixel_values'].squeeze(0))
        positive_inputs = self.tensor_transforms(self.processor(images=positive_img, return_tensors="pt")['pixel_values'].squeeze(0))
        negative_inputs = self.tensor_transforms(self.processor(images=negative_img, return_tensors="pt")['pixel_values'].squeeze(0))
        return anchor_inputs, positive_inputs, negative_inputs



# --- 2. Model Definition ---
class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DinoClassifier, self).__init__()
        self.dino = AutoModel.from_pretrained(DINOV2_SOURCE)
        hidden_size = self.dino.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.dino(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits



# --- UPDATED: Finetuning model now has a projection head and L2 normalization ---
class TripletDino(nn.Module):
    """Model for triplet loss finetuning with a projection head."""
    def __init__(self, embedding_dim=256, model_path=None):
        super().__init__()
        
        self.dino_version = getattr(config, 'DINO_VERSION', 'v2')
        self.is_finetuned = model_path is not None
        
        if self.dino_version == 'v2':
            _v2_src = getattr(config, 'DINOV2_MODEL_NAME', DINOV2_SOURCE)
            logging.info(f"Loading DINOv2 backbone: {_v2_src}")
            self.dino_base = AutoModel.from_pretrained(_v2_src)
            self.hidden_size = self.dino_base.config.hidden_size
        else:
            logging.info(f"Loading DINOv3 backbone via timm: {config.DINOV3_MODEL_NAME}")
            # Switching to timm to bypass HF gating issues
            self.dino_base = timm.create_model(config.DINOV3_MODEL_NAME, pretrained=True)
            self.hidden_size = self.dino_base.num_features 
                
        # --- UDA Freezing Strategy ---
        if getattr(config, 'FINETUNING_ADVANCED', {}).get("FREEZE_DINO_BACKBONE", False):
            logging.info(f"FREEZE_DINO_BACKBONE is True. Freezing the DINO{self.dino_version} backbone parameters.")
            for param in self.dino_base.parameters():
                param.requires_grad = False
                
        # Input is now hidden_size * 2 due to CLS + GAP concatenation
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # --- ROBUSTNESS FIX ---
        if self.is_finetuned:
            logging.info("Model initialized in FINETUNED mode (256-dim projection active).")
        else:
            logging.info("Model initialized in ZERO-SHOT mode (2048-dim raw features).")

    def set_finetuned_mode(self, mode: bool = True):
        """Allows dynamic switching of the projection head status."""
        self.is_finetuned = mode
        logging.info(f"Model mode manually set to: {'FINETUNED' if mode else 'ZERO-SHOT'}")

    def forward(self, pixel_values):
        if self.dino_version == 'v2':
            outputs = self.dino_base(pixel_values=pixel_values)
            # 1. Global context from [CLS] token
            cls_token = outputs.pooler_output # (batch_size, hidden_size)
            # 2. Local/texture info from patch tokens via Global Average Pooling (GAP)
            # DINOv2: [CLS, Patch1, Patch2, ...]
            patch_tokens = outputs.last_hidden_state[:, 1:, :] # (batch_size, seq_len-1, hidden_size)
            gap_features = torch.mean(patch_tokens, dim=1) # (batch_size, hidden_size)
        else:
            # 1. Global context & Patch context from timm model
            # forward_features returns [B, N, D] where N = CLS + Registers + Patches
            all_tokens = self.dino_base.forward_features(pixel_values)
            
            # DINOv3: [CLS, Reg1, Reg2, Reg3, Reg4, Patch1, Patch2, ...]
            # We skip index 0 (CLS) and index 1-4 (4 Register tokens) -> start from 5
            cls_token = all_tokens[:, 0, :]
            patch_tokens = all_tokens[:, 5:, :] # (batch_size, seq_len-5, hidden_size)
            gap_features = torch.mean(patch_tokens, dim=1)
        
        combined = torch.cat([cls_token, gap_features], dim=1) # (batch_size, hidden_size * 2)
        
        # Use projection head if we have pre-trained weights OR if we are currently training (learning from scratch)
        if self.is_finetuned or self.training:
            projected_embedding = self.projection_head(combined)
            # Apply L2 normalization for stable training with triplet loss
            return F.normalize(projected_embedding, p=2, dim=1)
        
        # Zero-Shot mode: return normalized raw features (2048-dim)
        return F.normalize(combined, p=2, dim=1)