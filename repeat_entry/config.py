import os
import torch
from datetime import datetime

# --- PROJECT ROOT ---
# Dynamic root for robustness
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # .../repeat_entry
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)                # .../ambient-machine

# --- MONGO ---
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://seawoods:sentinelMongo%40123@localhost:27017/sentinel?authSource=sentinel&replicaSet=rs0&directConnection=true"
)
DB_NAME = "sentinel"

# --- DYNAMIC DATE ---
# Automatically resolves to today's date (set at import time by the nightly runner).
# Can be overridden by setting the REPEAT_ENTRY_DATE env var (YYYY-MM-DD) for manual reruns.
_override_date = os.environ.get("REPEAT_ENTRY_DATE", "")
if _override_date:
    try:
        TODAY_DATE = datetime.strptime(_override_date, "%Y-%m-%d").date()
    except ValueError:
        TODAY_DATE = datetime.now().date()
else:
    TODAY_DATE = datetime.now().date()

TODAY_STR = TODAY_DATE.strftime("%Y-%m-%d")

# --- PIPELINE CONTROL ---
PIPELINE_STAGES = {
    "EXTRACT_EMBEDDINGS": True,
    "RUN_MATCHING_PIPELINE": True,
    "INTRA_DAY_ONLY": True
}
# Set to True to re-generate embeddings for EVERY track, even if they already exist in the .pkl.
FORCE_REEXTRACT = True 

# --- MODEL SETTINGS ---
# Set to True to bypass the custom MODEL_PATH and use the base DINOv3 backbone.
USE_BASE_MODEL = False

MODEL_PATH = r"C:\Users\nj.camera\Indriya\ambient-machine\src\models\best_model_loss.pth"
DINO_VERSION = 'v3'
DINOV3_MODEL_NAME = 'vit_large_patch16_dinov3'
DEVICE = "cpu"

# --- PERFORMANCE SETTINGS ---
# Multi-GPU batch size. V100 (16GB) handles ~128 well for DINOv3.
BATCH_SIZE = 128
NUM_WORKERS = 8

# --- DATA PATHS ---
# Local footfall root: repeat_entry/results/footfall/
# The nightly runner extracts crops here (YYYY-MM-DD sub-dirs), the extraction
# engine then iterates those sub-dirs automatically.
FOOTFALL_ROOT = os.path.join(PROJECT_ROOT, "results", "footfall")

# DYNAMIC NAMING: Automatically selects a separate file for the Base Model to avoid data corruption.
if USE_BASE_MODEL:
    EMBEDDINGS_PKL = os.path.join(PROJECT_ROOT, "data/embeddings/embeddings_fine_tuned_BASE.pkl")
else:
    EMBEDDINGS_PKL = os.path.join(PROJECT_ROOT, "data/embeddings/embeddings_fine_tuned.pkl")

# --- MATCHING RULES (Multi-Threshold) ---
MATCHING_RULES = [
    {"name": "Soft",      "sample_threshold": 0.73, "min_ratio": 0.40},
    {"name": "Hard",      "sample_threshold": 0.78, "min_ratio": 0.20},
    {"name": "Exception", "sample_threshold": 0.80, "min_ratio": 0.20}
]

# --- SAMPLING CONFIG ---
MAX_SAMPLES_PER_TRACK = 10
TRIM_BUFFER_SECONDS = 2.0

# --- GLOBAL QUALITY GATE ---
GLOBAL_MATCH_GATE = 0.90
GLOBAL_MATCH_MIN_SAMPLES = 2

# --- CENTROID QUALITY GATE ---
CENTROID_MATCH_THRESHOLD = 0.76

# --- BASE RESULTS DIRECTORY ---
BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
