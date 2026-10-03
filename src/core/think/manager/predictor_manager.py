# src/core/think/manager/predictor_manager.py

import os
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import torchvision.transforms as T
import torch.nn as nn

# Removed TRT imports as we are strictly CPU/.pt/.pth only

try:
    from ultralytics.nn.modules.block import C3k2  # noqa: F401
except Exception:
    pass

def _safe_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def _touch_parent_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

import sys
import subprocess

def _guess_rtdetr_repo_root(config_yaml_path: str) -> str:
    """
    Given .../rtdetrv2_pytorch/configs/xxxx.yml, return the repo root:
    .../rtdetrv2_pytorch
    """
    p = os.path.abspath(config_yaml_path)
    # Split at 'configs' folder if present
    parts = p.split(os.sep)
    if "configs" in parts:
        idx = parts.index("configs")
        return os.sep.join(parts[:idx])  # up to repo root (without trailing slash)
    # Fallback: go up two levels from the yaml file
    return os.path.dirname(os.path.dirname(p))

class PredictorManager:
    """
    Universal predictor that supports both YOLO and RT-DETR models.
    Loads the appropriate model based on modelName in config.
    """
    
    def __init__(
        self,
        model_config,
        device,
        logger,
        plot_overlays=False
    ):
        self.logger = logger
        self.device = device
        self.plot = plot_overlays
        self.model_config = model_config
        self.modelName = model_config.get('modelName', 'YOLO').upper()
        
        # Force CPU device
        self.device = 'cpu'
        self.logger.info("Forcing device to CPU.")
        
        # Initialize model based on type
        self._load_model()
        
        # Common attributes
        self.conf = model_config.get('conf', 0.6)
        self.class_ids = model_config.get('class_ids', [0])
        
    def _load_model(self):
        """Load the appropriate model based on modelName"""
        if self.modelName == 'YOLO':
            self._load_yolo_model()
        elif self.modelName == 'RT-DETR':
            self._load_rtdetr_model()
        else:
            raise ValueError(f"Unsupported modelName: {self.modelName}")
    
    def _load_yolo_model(self):
        """Load YOLO model (PyTorch or OpenVINO)."""
        model_path = self.model_config.get('model_path')
        tracker_path = self.model_config.get('tracker_path')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found at {model_path}")
        if not os.path.exists(tracker_path):
            raise FileNotFoundError(f"Tracker file not found at {tracker_path}")

        is_openvino = os.path.isdir(model_path) and "openvino" in model_path.lower()
        if not (model_path.endswith(".pt") or is_openvino):
            raise ValueError(f"Only .pt models or OpenVINO directories are supported. Received: {model_path}")

        self.backend = 'openvino' if is_openvino else 'pt'
        self.logger.info(f"Loading YOLO model: {model_path} on CPU (backend={self.backend})")
        
        self.model = YOLO(model_path, task="detect")

        self.model.tracker = tracker_path
        self.logger.info(f"YOLO model loaded successfully (backend={self.backend}).")
    
    def _load_rtdetr_model(self):
        """Load RT-DETR PyTorch model."""
        config_yaml_path = self.model_config.get('config_yaml_path')
        # Accept either 'model_path' or 'pth_checkpoint_path'
        pth_path = self.model_config.get('model_path') or self.model_config.get('pth_checkpoint_path')

        if not config_yaml_path or not pth_path:
            raise ValueError("RT-DETR requires 'config_yaml_path' and 'model_path' (pth).")
        
        if not (pth_path.endswith('.pt') or pth_path.endswith('.pth')):
            raise ValueError(f"Only .pt or .pth models are supported for RT-DETR. Received: {pth_path}")

        self.logger.info("Loading RT-DETR PyTorch model on CPU")
        self.model = self._load_rtdetr_pytorch(config_yaml_path, pth_path)
        self.rtdetr_mode = 'pytorch'
        self.backend = 'pt'

        # Common preprocessing
        self.rtdetr_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        self.logger.info(f"RT-DETR model loaded successfully (backend={self.backend}).")

    def _load_rtdetr_pytorch(self, yaml_path, ckpt_path):
        """Load RT-DETR PyTorch model from checkpoint"""
        cfg = YAMLConfig(yaml_path, resume=ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'ema' in checkpoint:
            state = checkpoint['ema'].get('module', checkpoint['ema'])
        else:
            state = checkpoint.get('model', checkpoint)
        
        cfg.model.load_state_dict(state)
        
        class Model(nn.Module):
            def __init__(self, _cfg):
                super().__init__()
                self.backbone = _cfg.model.deploy()
                self.postprocessor = _cfg.postprocessor.deploy()
            
            @torch.no_grad()
            def forward(self, images, orig_sizes):
                feats = self.backbone(images)
                outs = self.postprocessor(feats, orig_sizes)
                return outs
        
        return Model(cfg).to(self.device).eval()
    

    
    def predict(self, frame, current_time, run_time_str):
        """Universal predict method that routes to appropriate model"""
        if self.modelName == 'YOLO':
            return self._predict_yolo(frame, current_time, run_time_str)
        elif self.modelName == 'RT-DETR':
            return self._predict_rtdetr(frame, current_time, run_time_str)
        else:
            raise ValueError(f"Unsupported modelName: {self.modelName}")
    
    def _predict_yolo(self, frame, current_time, run_time_str):
        """YOLO prediction (existing logic)"""
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device
        )
        
        track_ids_dict = {}
        if not results or len(results) == 0 or results[0].boxes.id is None:
            return track_ids_dict, frame
        
        boxes = results[0].boxes.xyxy.int().cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        labels = results[0].boxes.cls.int().cpu().numpy()
        confidence = results[0].boxes.conf.cpu().numpy()
        class_names = results[0].names


        # self.logger.info(f"class_names : {class_names}")
        # self.logger.info(f"labels : {labels}")
        
        for idx, tid in enumerate(track_ids):
            updated_track_id = f"{run_time_str}_{tid:05d}"
            
            # DEBUG: Log the raw class ID to verify label mappings
            # self.logger.info(f"[DEBUG LABEL] Track ID: {updated_track_id} | Raw Class ID: {labels[idx]} | Mapped Name: {class_names[labels[idx]]}")

            track_ids_dict[updated_track_id] = {
                "track_id": updated_track_id,
                "bbox": boxes[idx],
                "label": labels[idx],
                "confidence": confidence[idx],
                "label_name": class_names[labels[idx]],
                "current_time": current_time
            }
        
        if self.plot:
            frame = results[0].plot(conf=False, labels=True, boxes=True, masks=True)
        
        return track_ids_dict, frame
    
    def _predict_rtdetr(self, frame, current_time, run_time_str):
        """RT-DETR prediction"""
        return self._predict_rtdetr_pytorch(frame, current_time, run_time_str)
    
    def _predict_rtdetr_pytorch(self, frame, current_time, run_time_str):
        """RT-DETR PyTorch prediction"""
        # Convert BGR to RGB and preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.rtdetr_transforms(img_rgb).unsqueeze(0).to(self.device)
        
        # Original frame size for postprocessing
        h, w, _ = frame.shape
        orig_size = torch.tensor([[w, h]], dtype=torch.float, device=self.device)
        
        # Run PyTorch inference
        labels, boxes, scores = self.model(img_tensor, orig_size)
        
        # Convert to TRT-like format for consistent processing
        outputs = {
            "labels": labels.detach().cpu().numpy(),
            "boxes": boxes.detach().cpu().numpy(),
            "scores": scores.detach().cpu().numpy()
        }
        
        return self._process_rtdetr_outputs(outputs, frame, current_time, run_time_str)

    def _process_rtdetr_outputs(self, outputs, frame, current_time, run_time_str):
        """
        Process RT-DETR outputs with robust handling of different output formats.
        """
        try:
            def _to_numpy(x):
                """Safely convert tensor/array to numpy array"""
                if x is None:
                    return None
                if isinstance(x, (list, tuple)) and len(x) == 1:
                    x = x[0]
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                if isinstance(x, np.ndarray):
                    return x
                try:
                    return np.asarray(x)
                except:
                    return None

            labels_np = boxes_np = scores_np = None

            # Handle different output formats
            if isinstance(outputs, dict):
                # Dictionary output - find relevant keys
                for key, value in outputs.items():
                    key_lower = str(key).lower()
                    if any(term in key_lower for term in ['boxes', 'bbox']):
                        boxes_np = _to_numpy(value)
                    elif any(term in key_lower for term in ['labels', 'label', 'cls', 'class']):
                        labels_np = _to_numpy(value)
                    elif any(term in key_lower for term in ['scores', 'score', 'conf', 'confidence']):
                        scores_np = _to_numpy(value)
                        
            elif isinstance(outputs, (list, tuple)):
                # List/tuple output - identify by shape and dtype
                arrays = [_to_numpy(o) for o in outputs if o is not None]
                arrays = [a for a in arrays if isinstance(a, np.ndarray)]
                
                # Find boxes (should have last dimension = 4)
                for i, arr in enumerate(arrays):
                    if arr.ndim >= 2 and arr.shape[-1] == 4:
                        boxes_np = arr
                        break
                
                # Find labels (should be integer dtype)
                for i, arr in enumerate(arrays):
                    if (arr is not boxes_np and isinstance(arr, np.ndarray) 
                        and np.issubdtype(arr.dtype, np.integer)):
                        labels_np = arr
                        break
                
                # Find scores (remaining array)
                for i, arr in enumerate(arrays):
                    if arr is not boxes_np and arr is not labels_np:
                        scores_np = arr
                        break
                        
            elif torch.is_tensor(outputs) or isinstance(outputs, np.ndarray):
                # Single tensor/array (e.g., Nx6 format: x1,y1,x2,y2,score,label)
                arr = _to_numpy(outputs)
                if arr is not None and arr.ndim == 2 and arr.shape[1] >= 6:
                    boxes_np = arr[:, 0:4]
                    scores_np = arr[:, 4]
                    labels_np = arr[:, 5].astype(np.int32)

            # Validate we have the required arrays
            if boxes_np is None or scores_np is None or labels_np is None:
                self.logger.warning("Could not extract boxes, scores, and labels from RT-DETR outputs")
                return {}, frame

            # Clean up dimensions
            if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
                boxes_np = np.reshape(boxes_np, (-1, 4))
            scores_np = np.squeeze(scores_np)
            labels_np = np.squeeze(labels_np).astype(np.int32)

            # Apply confidence and class filtering
            if len(scores_np.shape) > 0:  # Handle case where scores might be scalar
                conf_mask = scores_np > float(self.conf)
            else:
                conf_mask = np.array([scores_np > float(self.conf)])
                
            if self.class_ids:
                class_mask = np.isin(labels_np, np.array(self.class_ids, dtype=np.int32))
                mask = conf_mask & class_mask
            else:
                mask = conf_mask

            if not np.any(mask):
                return {}, frame

            # Filter arrays
            boxes_np = boxes_np[mask]
            scores_np = scores_np[mask] if scores_np.ndim > 0 else np.array([scores_np])
            labels_np = labels_np[mask] if labels_np.ndim > 0 else np.array([labels_np])

            # Build track dictionary
            track_ids_dict = {}
            for idx, (box, score, label) in enumerate(zip(boxes_np, scores_np, labels_np)):
                track_id = f"{run_time_str}_{idx:05d}"
                track_ids_dict[track_id] = {
                    "track_id": track_id,
                    "bbox": box.astype(int),
                    "label": int(label),
                    "confidence": float(score),
                    "label_name": f"class_{int(label)}",
                    "current_time": current_time,
                }

            if self.plot:
                frame = self._plot_rtdetr_detections(frame, track_ids_dict)

            return track_ids_dict, frame

        except Exception as e:
            self.logger.error(f"Error processing RT-DETR outputs: {e}")
            return {}, frame
    
    def _plot_rtdetr_detections(self, frame, track_ids_dict):
        """Plot RT-DETR detections on frame"""
        for track_info in track_ids_dict.values():
            bbox = track_info["bbox"]
            conf = track_info["confidence"]
            label = track_info["label_name"]
            track_id = track_info["track_id"]
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}:{track_id}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame


    
    def finish(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up UniversalPredictor resources...")
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("UniversalPredictor resources released.")