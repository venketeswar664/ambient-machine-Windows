import os
import sys
import pickle
import torch
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from sentence_transformers.util import cos_sim
from pymongo import MongoClient, UpdateOne

# Handle pathing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import config as local_config
from src.utils.visualizer import save_detailed_repeat_audit_collage


def _parse_day_to_datetime(day_str: str) -> datetime:
    """
    Converts a 'YYYY-MM-DD' string to a timezone-naive datetime at midnight.
    This matches the MongoDB Date format shown in the schema.
    """
    try:
        return datetime.strptime(day_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _raw_track_id(full_key: str) -> str:
    """
    Extract the bare track_id from an embedding key.

    Embedding key format:
        '{day}/{YYYY-MM-DD_HH-MM-SS}_{track_id}'
    Example:
        '2026-06-26/2026-06-26_07-00-49_26062026043105737248_00629'
        -> '26062026043105737248_00629'

    The timestamp prefix is always exactly 19 chars (YYYY-MM-DD_HH-MM-SS)
    followed by a single underscore, so we skip 20 chars after the '/'.
    """
    try:
        track_folder = full_key.split('/', 1)[1]   # drop the 'YYYY-MM-DD/' prefix
        return track_folder[20:]                    # skip 'YYYY-MM-DD_HH-MM-SS_'
    except (IndexError, ValueError):
        return full_key


def push_results_to_db(json_results: list, logger) -> None:
    """
    Upserts repeat-entry results into the `trackid_mapping` collection.

    Schema (mirrors user-confirmed format):
        {
            day          : ISODate (midnight of the day),
            track_id     : str,
            identity_id  : str,
            visit_number : int,
            is_repeat    : bool,
            avg_similarity: float,
            verdict_rule : str | null,
            startTime    : ISODate
        }

    Upsert key: (track_id, day) — safe to re-run nightly without duplicating.
    """
    if not json_results:
        logger.info("push_results_to_db: no results to push, skipping.")
        return

    try:
        client = MongoClient(local_config.MONGO_URI, serverSelectionTimeoutMS=10_000)
        db = client[local_config.DB_NAME]
        collection = db["trackid_mapping"]

        ops = []
        for rec in json_results:
            day_dt = _parse_day_to_datetime(rec.get("day", ""))

            # Parse startTime — may already be an isoformat string
            start_raw = rec.get("startTime")
            if isinstance(start_raw, str):
                try:
                    start_dt = datetime.fromisoformat(start_raw)
                except ValueError:
                    start_dt = None
            elif isinstance(start_raw, datetime):
                start_dt = start_raw
            else:
                start_dt = None

            doc = {
                "day"           : day_dt,
                "track_id"      : rec.get("track_id"),
                "identity_id"   : rec.get("identity_id"),
                "visit_number"  : rec.get("visit_number", 1),
                "is_repeat"     : rec.get("is_repeat", False),
                "avg_similarity": rec.get("avg_similarity", 0.0),
                "verdict_rule"  : rec.get("verdict_rule"),
                "startTime"     : start_dt,
            }

            ops.append(
                UpdateOne(
                    filter={"track_id": doc["track_id"], "day": doc["day"]},
                    update={"$set": doc},
                    upsert=True,
                )
            )

        if ops:
            result = collection.bulk_write(ops, ordered=False)
            logger.info(
                f"📦 DB push complete — "
                f"upserted: {result.upserted_count}, "
                f"modified: {result.modified_count}, "
                f"total: {len(ops)}"
            )

    except Exception as e:
        logger.error(f"❌ push_results_to_db failed: {e}", exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


def get_track_timestamp(track_items):
    from src.utils.visualizer import parse_footfall_timestamp
    timestamps = []
    for path, _ in track_items:
        ts = parse_footfall_timestamp(path)
        if ts:
            timestamps.append(ts)
    return min(timestamps) if timestamps else datetime.min


def run_matching(logger, output_dir):

    logger.info("--- Stage: Vectorized Identity Matching & Audit ---")

    if not os.path.exists(local_config.EMBEDDINGS_PKL):
        logger.error(f"❌ Embeddings missing: {local_config.EMBEDDINGS_PKL}")
        return pd.DataFrame()

    with open(local_config.EMBEDDINGS_PKL, 'rb') as f:
        all_embeddings = pickle.load(f)

    track_info = []
    for tid in all_embeddings.keys():
        t_day = tid.split('/')[0]
        if t_day != local_config.TODAY_STR:  # Only match tracks for the configured target date
            continue
        track_info.append({
            'id': tid,
            'day': t_day,
            'timestamp': get_track_timestamp(all_embeddings[tid])
        })

    sorted_tracks = sorted(track_info, key=lambda x: x['timestamp'])
    logger.info(f"📋 Loaded {len(sorted_tracks)} tracks for matching.")

    enrolled_ids = []
    enrolled_days = []
    enrolled_timestamps = []
    gallery_track_indices = []
    gallery_tensor = None

    audit_registry = {}
    results = []

    identity_map = {}
    identity_visit_count = {}
    json_results = []

    last_day = None

    for track_idx, track_data in enumerate(tqdm(sorted_tracks, desc="Identity Search")):

        current_id = track_data['id']
        current_day = track_data['day']
        current_timestamp = track_data['timestamp']

        if current_day != last_day:
            logger.info(f"📅 Entering Day Partition: {current_day}")

            if local_config.PIPELINE_STAGES.get("INTRA_DAY_ONLY", True):
                enrolled_ids = []
                enrolled_days = []
                enrolled_timestamps = []
                gallery_tensor = None
                gallery_track_indices = []

            last_day = current_day

        items = all_embeddings[current_id]
        current_paths, current_embs_list = zip(*items)

        current_embs = torch.stack(current_embs_list).to(local_config.DEVICE)
        current_centroid = torch.mean(current_embs, dim=0).unsqueeze(0)

        is_repeat = False
        match_info = {}
        sample_scores = []

        if gallery_tensor is not None:

            sim_matrix = cos_sim(current_embs, gallery_tensor)

            for gallery_idx in range(len(enrolled_ids)):

                track_mask = torch.tensor(
                    [i == gallery_idx for i in gallery_track_indices],
                    device=local_config.DEVICE
                )

                if not torch.any(track_mask):
                    continue

                track_sims = sim_matrix[:, track_mask]
                max_per_current, _ = torch.max(track_sims, dim=1)

                # ✅ STORE REAL SIMILARITY HERE
                sample_scores = max_per_current.detach().cpu().tolist()

                # RULE LOGIC
                rule_pass = {r['name'].lower(): False for r in local_config.MATCHING_RULES}

                for rule in local_config.MATCHING_RULES:
                    ratio = torch.sum(max_per_current >= rule['sample_threshold']).item() / len(current_embs)
                    if ratio >= rule['min_ratio']:
                        rule_pass[rule['name'].lower()] = True

                gate_hits = torch.sum(max_per_current >= local_config.GLOBAL_MATCH_GATE).item()
                gate_ok = gate_hits >= local_config.GLOBAL_MATCH_MIN_SAMPLES

                e_embs = gallery_tensor[track_mask]
                e_centroid = torch.mean(e_embs, dim=0).unsqueeze(0)

                c_sim = torch.nn.functional.cosine_similarity(current_centroid, e_centroid).item()
                pass_centroid = c_sim >= local_config.CENTROID_MATCH_THRESHOLD

                if gate_ok or (rule_pass['soft'] and (rule_pass['hard'] or rule_pass['exception']) and pass_centroid):

                    is_repeat = True

                    # ✅ STORE TOP MATCH SAMPLES
                    topk = torch.topk(max_per_current, k=min(5, len(max_per_current)))

                    verdict_samples = [
                        {
                            "path": current_paths[i],
                            "score": max_per_current[i].item()
                        }
                        for i in topk.indices
                    ]

                    match_info = {
                        'repeat_of': enrolled_ids[gallery_idx],
                        'repeat_of_day': enrolled_days[gallery_idx],
                        'avg_sim': torch.mean(max_per_current).item(),
                        'verdict': "Gate-Pass" if gate_ok else "Logic-Pass",
                        'sample_scores': sample_scores,
                        'verdict_samples': verdict_samples
                    }

                    break

        # 🔁 REPEAT CASE
        if is_repeat:

            base_identity = identity_map.get(match_info['repeat_of'], None)

            if base_identity is None:
                base_identity = f"ID_{len(identity_map)}"
                identity_map[match_info['repeat_of']] = base_identity
                identity_visit_count[base_identity] = 1

            identity_map[current_id] = base_identity
            identity_visit_count[base_identity] += 1

            visit_number = identity_visit_count[base_identity]

            results.append({
                'day': current_day,
                'track_id': current_id,
                'timestamp': current_timestamp,
                'is_repeat': True,
                'repeat_of': match_info['repeat_of'],
                'repeat_of_day': match_info['repeat_of_day'],
                'avg_similarity': match_info['avg_sim'],
                'verdict_rule': match_info['verdict'],
                'sample_scores': match_info['sample_scores'],
                'verdict_samples': match_info['verdict_samples']
            })

            json_results.append({
                "day": current_day,
                "track_id": _raw_track_id(current_id),
                "identity_id": identity_map[current_id],
                "visit_number": identity_visit_count[identity_map[current_id]],
                "is_repeat": True,
                "avg_similarity": match_info['avg_sim'],
                "verdict_rule": match_info['verdict'],
                "startTime": current_timestamp.isoformat(timespec='seconds') if current_timestamp else None
            })

        else:
            identity_id = f"ID_{len(identity_map)}"
            identity_map[current_id] = identity_id
            identity_visit_count[identity_id] = 1

            enrolled_ids.append(current_id)
            enrolled_days.append(current_day)
            enrolled_timestamps.append(current_timestamp)

            gallery_tensor = torch.cat([gallery_tensor, current_embs], dim=0) if gallery_tensor is not None else current_embs
            gallery_track_indices.extend([len(enrolled_ids)-1] * len(current_embs))

            results.append({
                'day': current_day,
                'track_id': current_id,
                'timestamp': current_timestamp,
                'is_repeat': False,
                'repeat_of': None,
                'repeat_of_day': None,
                'avg_similarity': 0.0,
                'verdict_rule': None,
                'sample_scores': [],
                'verdict_samples': []
            })

            json_results.append({
                "day": current_day,
                "track_id": _raw_track_id(current_id),
                "identity_id": identity_map[current_id],
                "visit_number": 1,
                "is_repeat": False,
                "avg_similarity": 0.0,
                "verdict_rule": None,
                "startTime": current_timestamp.isoformat(timespec='seconds') if current_timestamp else None
            })

        audit_registry[current_id] = {
            'track_id': current_id,
            'day': current_day,
            'timestamp': current_timestamp,
            'images': current_paths
        }

    # ✅ CREATE DF
    df = pd.DataFrame(results)

    logger.info(f"📊 DataFrame created with columns: {df.columns}")

    # ✅ VISUALIZATION FIX
    for idx, row in df.iterrows():
        if not row['is_repeat']:
            continue

        day_folder = os.path.join(output_dir, row['day'], "repeat_entry_collages")
        os.makedirs(day_folder, exist_ok=True)

        enroll_data = audit_registry.get(row['repeat_of'])

        if enroll_data:  

            current_visit_data = {
                'track_id': row['track_id'],
                'day': row['day'],
                'timestamp': row['timestamp'],
                'images': [p for p, _ in all_embeddings[row['track_id']]],
                'sample_scores': row['sample_scores'],
                'verdict_samples': row['verdict_samples'],
                'verdict_rule': row['verdict_rule']
            }

            save_detailed_repeat_audit_collage(
                enrollment_data=enroll_data,
                repeat_data=current_visit_data,
                output_dir=day_folder,
                threshold=0.70,
                logger=logger
            )

    logger.info("🎉 Pipeline Execution Complete")

    # ✅ SAVE METADATA JSON
    metadata_path = os.path.join(output_dir, "track_metadata.json")

    try:
        with open(metadata_path, "w") as f:
            json.dump(json_results, f, indent=4)
        
        logger.info(f"📄 track_metadata.json saved at: {metadata_path}")

    except Exception as e:
        logger.error(f"❌ Failed to save track_metadata.json: {e}", exc_info=True)

    # ✅ PUSH TO MONGODB (trackid_mapping)
    push_results_to_db(json_results, logger)

    return df

# import os
# import sys
# import pickle
# import torch
# import pandas as pd
# from datetime import datetime
# from tqdm import tqdm
# from sentence_transformers.util import cos_sim
# import numpy as np

# # Handle pathing for submodules
# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # Package imports
# import config as local_config
# from src.utils.helpers import parse_timestamp_from_path
# from src.utils.visualizer import save_detailed_repeat_audit_collage

# def get_track_timestamp(track_items):
#     """Get the earliest timestamp from a list of (path, embedding) tuples."""
#     from src.utils.visualizer import parse_footfall_timestamp
#     timestamps = []
#     for path, _ in track_items:
#         ts = parse_footfall_timestamp(path)
#         if ts:
#             timestamps.append(ts)
#     return min(timestamps) if timestamps else datetime.min

# def run_matching(logger, output_dir):
#     """
#     High-performance vectorized matching engine with full visual audit restoration.
#     """
#     logger.info("--- Stage: Vectorized Identity Matching & Audit ---")
    
#     if not os.path.exists(local_config.EMBEDDINGS_PKL):
#         logger.error(f"❌ Embeddings missing: {local_config.EMBEDDINGS_PKL}")
#         return pd.DataFrame()
    
#     with open(local_config.EMBEDDINGS_PKL, 'rb') as f:
#         all_embeddings = pickle.load(f)
    
#     track_info = []
#     for tid in all_embeddings.keys():
#         t_day = tid.split('/')[0]
#         track_info.append({
#             'id': tid, 'day': t_day, 'timestamp': get_track_timestamp(all_embeddings[tid])
#         })
    
#     sorted_tracks = sorted(track_info, key=lambda x: x['timestamp'])
#     logger.info(f"📋 Loaded {len(sorted_tracks)} tracks for matching.")
    
#     enrolled_ids = []
#     enrolled_days = []
#     enrolled_timestamps = []
#     gallery_track_indices = [] 
    
#     # Persistent registry for visual audit lookups (doesn't reset daily)
#     audit_registry = {}
    
#     results = []
#     stats = {
#         "gate_pass": 0, 
#         "centroid_pass": 0, 
#         "total_repeats": 0
#     }
#     # Initialize rule stats dynamically from config
#     for rule in local_config.MATCHING_RULES:
#         stats[f"{rule['name'].lower()}_pass"] = 0

#     last_day = None
#     for track_idx, track_data in enumerate(tqdm(sorted_tracks, desc="Identity Search")):
#         current_id = track_data['id']
#         current_day = track_data['day']
#         current_timestamp = track_data['timestamp']
        
#         if current_day != last_day:
#             logger.info(f"📅 Entering Day Partition: {current_day} (Gallery: {len(enrolled_ids)})")
            
#             # Reset gallery for Intra-Day logic
#             if local_config.PIPELINE_STAGES.get("INTRA_DAY_ONLY", True):
#                 if last_day is not None:
#                     logger.info(f"🔄 Resetting gallery for new day: {current_day}")
#                 enrolled_ids = []
#                 enrolled_days = []
#                 enrolled_timestamps = []
#                 enrolled_paths_list = []
#                 gallery_tensor = None
#                 gallery_track_indices = []
                
#             last_day = current_day

#         items = all_embeddings[current_id]
#         current_paths, current_embs_list = zip(*items)
#         current_embs = torch.stack(current_embs_list).to(local_config.DEVICE)
#         current_centroid = torch.mean(current_embs, dim=0).unsqueeze(0)
        
#         is_repeat = False
#         match_info = {}
        
#         if gallery_tensor is not None:
#             # 1. High-Speed Population Matrix Comparison
#             sim_matrix = cos_sim(current_embs, gallery_tensor)
            
#             # 2. Iterate Gallery to find best match
#             for gallery_idx in range(len(enrolled_ids)):
#                 track_mask = torch.tensor([i == gallery_idx for i in gallery_track_indices], device=local_config.DEVICE)
#                 if not torch.any(track_mask): continue
                
#                 track_sims = sim_matrix[:, track_mask]
#                 max_per_current, _ = torch.max(track_sims, dim=1)
                
#                 # Evaluation Logic
#                 rule_pass = {r['name'].lower(): False for r in local_config.MATCHING_RULES}
#                 for rule in local_config.MATCHING_RULES:
#                     ratio = torch.sum(max_per_current >= rule['sample_threshold']).item() / len(current_embs)
#                     if ratio >= rule['min_ratio']:
#                         rule_pass[rule['name'].lower()] = True
#                         stats[f"{rule['name'].lower()}_pass"] += 1
                
#                 gate_hits = torch.sum(max_per_current >= local_config.GLOBAL_MATCH_GATE).item()
#                 gate_ok = gate_hits >= local_config.GLOBAL_MATCH_MIN_SAMPLES
#                 if gate_ok: stats["gate_pass"] += 1
                
#                 # Centroid Check
#                 e_embs = gallery_tensor[track_mask]
#                 e_centroid = torch.mean(e_embs, dim=0).unsqueeze(0)
#                 c_sim = torch.nn.functional.cosine_similarity(current_centroid, e_centroid).item()
#                 pass_centroid = c_sim >= local_config.CENTROID_MATCH_THRESHOLD
#                 if pass_centroid: stats["centroid_pass"] += 1
                
#                 # Hybrid Decision
#                 if gate_ok or (rule_pass['soft'] and (rule_pass['hard'] or rule_pass['exception']) and pass_centroid):
#                     is_repeat = True
#                     match_info = {
#                         'repeat_of': enrolled_ids[gallery_idx], 
#                         'repeat_of_day': enrolled_days[gallery_idx],
#                         'avg_sim': torch.mean(max_per_current).item(),
#                         'verdict': "Gate-Pass" if gate_ok else "Logic-Pass",
#                         'sample_scores': max_per_current.cpu().tolist(),
#                         'indices': (max_per_current >= 0.90).nonzero(as_tuple=True)[0].cpu().tolist()
#                     }
#                     stats['total_repeats'] += 1
#                     break

#         if is_repeat:
#             results.append({
#                 'day': current_day, 'track_id': current_id, 'timestamp': current_timestamp,
#                 'is_repeat': True, 'repeat_of': match_info['repeat_of'], 'repeat_of_day': match_info['repeat_of_day'],
#                 'avg_similarity': match_info['avg_sim'], 'verdict_rule': match_info['verdict'],
#                 'sample_scores': match_info['sample_scores'],
#                 'verdict_samples': [{'path': current_paths[i], 'score': match_info['sample_scores'][i]} for i in match_info['indices'][:10]]
#             })
#             logger.info(f"✅ REPEAT: {current_id} -> {match_info['repeat_of']} ({match_info['verdict']})")
#         else:
#             # Enroll
#             enrolled_ids.append(current_id)
#             enrolled_days.append(current_day)
#             enrolled_timestamps.append(current_timestamp)
#             enrolled_paths_list.append(current_paths)
#             gallery_tensor = torch.cat([gallery_tensor, current_embs], dim=0) if gallery_tensor is not None else current_embs
#             gallery_track_indices.extend([len(enrolled_ids)-1] * len(current_embs))
            
#             results.append({
#                 'day': current_day, 'track_id': current_id, 'timestamp': current_timestamp,
#                 'is_repeat': False, 'repeat_of': None, 'repeat_of_day': None,
#                 'avg_similarity': 0.0, 'verdict_rule': None, 'verdict_samples': [], 'sample_scores': []
#             })
            
#         # Update Audit Registry (Persists across day resets)
#         audit_registry[current_id] = {
#             'track_id': current_id, 'day': current_day, 'timestamp': current_timestamp, 'images': current_paths
#         }
            
#     # --- Post-Matching: Audit Collage Generation ---
#     df = pd.DataFrame(results)
#     csv_p = os.path.join(output_dir, "identity_results.csv")
#     df.drop(columns=['sample_scores', 'verdict_samples']).to_csv(csv_p, index=False)
    
#     logger.info("🎨 Generating visual audit proofs for detected repeats...")
    
#     # Get threshold from first rule (usually Soft)
#     soft_threshold = local_config.MATCHING_RULES[0]['sample_threshold'] if local_config.MATCHING_RULES else 0.91

#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Collage Generation"):
#         if not row['is_repeat']: continue
        
#         day_folder = os.path.join(output_dir, row['day'], "repeat_entry_collages")
#         os.makedirs(day_folder, exist_ok=True)
        
#         enroll_data = audit_registry.get(row['repeat_of'])
#         if enroll_data:
#             current_visit_data = {
#                 'track_id': row['track_id'], 'day': row['day'], 'timestamp': row['timestamp'],
#                 'images': [p for p, _ in all_embeddings[row['track_id']]],
#                 'sample_scores': row['sample_scores'], 'verdict_rule': row['verdict_rule'],
#                 'verdict_samples': row['verdict_samples']
#             }
#             save_detailed_repeat_audit_collage(
#                 enrollment_data=enroll_data, repeat_data=current_visit_data,
#                 output_dir=day_folder, threshold=soft_threshold, logger=logger
#             )
            
#     # --- Final Pipeline Summary ---
#     logger.info("--- Final Pipeline Summary ---")
#     for rule in local_config.MATCHING_RULES:
#         rule_key = f"{rule['name'].lower()}_pass"
#         logger.info(f"Condition '{rule['name']}': {stats[rule_key]} triggers")
#     logger.info(f"Condition 'Global Gate': {stats['gate_pass']} triggers")
#     logger.info(f"Condition 'Centroid Matching': {stats['centroid_pass']} triggers")
#     logger.info(f"TOTAL REPEAT ENTRIES IDENTIFIED: {stats['total_repeats']}")
#     logger.info("-----------------------------------")

#     logger.info(f"🎉 Pipeline Execution Complete. FINAL MATCHES: {stats['total_repeats']}")
#     return df

# if __name__ == "__main__":
#     from src.utils.helpers import setup_logging
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_dir = os.path.join(local_config.BASE_RESULTS_DIR, f"run_{run_id}")
#     os.makedirs(out_dir, exist_ok=True)
#     logger = setup_logging(out_dir, "standalone_match")
#     run_matching(logger, out_dir)
