import os
import re
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_run_{run_id}.log")
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)-8s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
    return logger

def get_latest_archive_paths(base_dir: str, num_paths: int = 2) -> List[Path]:
    # parent_path = Path(base_dir)
    # if not parent_path.is_dir(): return []
    archive_dirs = [Path(p) for p in base_dir]
    return sorted(archive_dirs, key=lambda p: p.name, reverse=True)[:num_paths]

def get_start_time_from_archive(path: Path) -> Optional[datetime]:
    match = re.search(r'(\d{8}T\d{6}Z)', path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    return None

def parse_timestamp_from_path(path: str) -> Optional[datetime]:
    stem = Path(path).stem
    
    # Format 1: YYYY-MM-DD HH:MM:SS.mmm
    match = re.search(r'(\d{4}-\d{2}-\d{2}[ _]\d{2}[:\-]\d{2}[:\-]\d{2}\.\d{3})', stem)
    if match:
        try:
            val = match.group(1).replace('-', ':').replace('_', ' ')
            # Fix date part
            val = val[:10].replace(':', '-') + val[10:]
            return datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            pass

    # Format 2: DDMMYYYYHHMMSS
    # Typically 03042026043049...
    if len(stem) >= 14 and stem[:14].isdigit():
        try:
            return datetime.strptime(stem[:14], "%d%m%Y%H%M%S")
        except ValueError:
            pass
            
    return None