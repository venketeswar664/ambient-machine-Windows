import os
import sys
from datetime import datetime

# Force CPU mode globally by hiding CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Handle pathing
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Module imports
import config as local_config
from src.extraction.engine import run_extraction
from src.matching.pipeline import run_matching
from src.utils.telemetry import tracker
from src.utils.helpers import setup_logging

def main():
    """
    Unified entry point for the Repeat Entry Detection Pipeline.
    Coordinates extraction and matching with full telemetry and resource reporting.
    """
    # 1. Setup dynamic output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(local_config.BASE_RESULTS_DIR, f"run_{run_id}")
    os.makedirs(run_folder, exist_ok=True)
    
    # 2. Setup unified logging in the run folder
    logger = setup_logging(run_folder, f"repeat_entry_main")
    
    print("\n" + "="*50)
    print("🚀 UNIFIED REPEAT ENTRY DETECTION PIPELINE (V2)")
    print("="*50)
    print(f"🔹 Run ID: {run_id}")
    print(f"🔹 Output: {run_folder}")
    print(f"🔹 Device: {local_config.DEVICE}")
    print("-"*50)

    logger.info("==========================================")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output Directory: {run_folder}")
    
    # 3. Stage: Embedding Extraction
    if local_config.PIPELINE_STAGES.get("EXTRACT_EMBEDDINGS", False):
        try:
            with tracker.stage("Embedding Extraction", logger):
                run_extraction(logger, run_folder)
        except Exception as e:
            logger.error(f"❌ ERROR in Extraction: {e}", exc_info=True)
            return
    else:
        logger.info("Stage: Embedding Extraction skipped (per config).")
        
    # 4. Stage: Matching Pipeline
    if local_config.PIPELINE_STAGES.get("RUN_MATCHING_PIPELINE", False):
        try:
            with tracker.stage("Identity Matching & Audit", logger):
                run_matching(logger, run_folder)
        except Exception as e:
            logger.error(f"❌ ERROR in Matching: {e}", exc_info=True)
    else:
        logger.info("Stage: Matching Pipeline skipped (per config).")

    # 5. Final Performance & Resource Report
    perf_summary = tracker.get_summary_report()
    print(perf_summary)
    logger.info(perf_summary)

    print("\n" + "="*50)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print(f"📂 Results archived in: {run_folder}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

