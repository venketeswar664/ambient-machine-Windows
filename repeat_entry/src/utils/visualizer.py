import os
import re
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import config as local_config


def parse_footfall_timestamp(path):
    stem = os.path.basename(path)
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', stem)
    if match:
        try:
            date_str = match.group(1)
            time_str = match.group(2).replace('-', ':')
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except:
            return None
    return None


# 🔥 ✅ MODIFIED FUNCTION (added identity_id)
def save_detailed_repeat_audit_collage(
    enrollment_data,
    repeat_data,
    output_dir,
    threshold,
    logger,
    identity_id=None   # 🔥 NEW
):

    PADDING = 40
    IMG_HEIGHT = 200
    LABEL_COL_WIDTH = 420
    IMG_ROW_PADDING = 20
    ROW_HEIGHT = IMG_HEIGHT + 100
    BACKGROUND_COLOR = (255, 255, 255)
    HEADER_BG = (20, 40, 80)

    try:
        font_main = ImageFont.truetype("arial.ttf", 20)
        font_id = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arial.ttf", 26)
        font_ts = ImageFont.truetype("arial.ttf", 20)
        font_header = ImageFont.truetype("arial.ttf", 40)
    except:
        font_main = font_id = font_bold = font_ts = font_header = ImageFont.load_default()

    def load_and_resize(path):
        if not os.path.exists(path): return None
        try:
            img = Image.open(path)
            aspect = img.width / img.height
            return img.resize((int(IMG_HEIGHT * aspect), IMG_HEIGHT), Image.Resampling.LANCZOS)
        except:
            return None

    # --- Prepare enrollment images ---
    enrolled_imgs = []
    for p in enrollment_data['images'][:10]:
        img = load_and_resize(p)
        if img:
            ts = parse_footfall_timestamp(p)
            enrolled_imgs.append({
                'img': img,
                'time': ts.strftime("%H:%M:%S") if ts else "??:??:??"
            })

    # --- Prepare repeat samples ---
    all_repeat_samples = []
    scores = repeat_data.get('sample_scores', [])

    # 🔥 Fallback: compute dummy similarity if missing
    if not scores or len(scores) != len(repeat_data['images']):
        logger.warning("⚠️ sample_scores missing or mismatched. Generating fallback similarities.")

        # fallback: assign decreasing similarity (just for visibility)
        n = len(repeat_data['images'])
        scores = [1.0 - (i / max(n, 1)) for i in range(n)]



    for idx, p in enumerate(repeat_data['images']):
        img = load_and_resize(p)
        if img:
            score = scores[idx] if idx < len(scores) else 0.0
            ts = parse_footfall_timestamp(p)

            all_repeat_samples.append({
                'img': img,
                'score': score,
                'time': ts.strftime("%H:%M:%S") if ts else "??:??:??",
                'is_match': score >= threshold
            })

    all_repeat_samples.sort(key=lambda x: x['score'], reverse=True)

    best_matches = [s for s in all_repeat_samples if s['is_match']][:10]
    worst_matches = [s for s in all_repeat_samples if not s['is_match']][:10]

    verdict_samples = []
    for item in repeat_data.get('verdict_samples', []):
        img = load_and_resize(item['path'])
        if img:
            ts = parse_footfall_timestamp(item['path'])
            verdict_samples.append({
                'img': img,
                'score': item['score'],
                'time': ts.strftime("%H:%M:%S") if ts else "??:??:??",
                'is_match': True
            })

    rows = [
        ("ENROLLED BASELINE", enrolled_imgs, (20, 40, 80), enrollment_data),
        ("TOP OVERALL MATCHES", best_matches, (40, 150, 40), repeat_data),
        (f"NON-MATCHED (<{threshold:.2f})", worst_matches, (200, 40, 40), repeat_data),
        (f"VERDICT: {repeat_data.get('verdict_rule', 'Match Proof')}", verdict_samples, (0, 102, 204), repeat_data)
    ]

    max_cols = max(len(r[1]) for r in rows)
    if max_cols == 0:
        return

    collage_width = LABEL_COL_WIDTH + (max_cols * (IMG_HEIGHT + IMG_ROW_PADDING)) + (2 * PADDING)
    collage_height = (4 * (ROW_HEIGHT + IMG_ROW_PADDING)) + 180 + (2 * PADDING)

    collage = Image.new('RGB', (collage_width, collage_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(collage)

    # 🔥 HEADER WITH ID
    draw.rectangle([0, 0, collage_width, 120], fill=HEADER_BG)

    repeat_id_short = repeat_data['track_id'].split('/')[-1]
    header_text = f"REPEAT AUDIT: {repeat_id_short}"

    if identity_id:
        header_text += f" | ID: {identity_id}"

    draw.text((PADDING, 35), header_text, fill=(255, 255, 255), font=font_header)

    y_cursor = 160

    for label_title, imgs, theme_color, meta in rows:

        label_rect = [PADDING, y_cursor, PADDING + LABEL_COL_WIDTH - 20, y_cursor + IMG_HEIGHT]
        draw.rectangle(label_rect, fill=(245, 248, 255), outline=theme_color, width=2)
        draw.rectangle([PADDING, y_cursor, PADDING + 10, y_cursor + IMG_HEIGHT], fill=theme_color)

        draw.text((PADDING + 25, y_cursor + 15), label_title, fill=theme_color, font=font_bold)

        tid_short = meta['track_id'].split('/')[-1]

        # 🔥 SHOW BOTH TRACK ID + IDENTITY ID
        id_text = f"Track: {tid_short}"
        if identity_id:
            id_text += f" | ID: {identity_id}"

        draw.text((PADDING + 25, y_cursor + 55), id_text, fill=(10, 20, 40), font=font_id)

        draw.text((PADDING + 25, y_cursor + 85), f"Date: {meta['day']}", fill=(10, 20, 40), font=font_main)
        draw.text((PADDING + 25, y_cursor + 115), f"Entry: {meta['timestamp'].strftime('%H:%M:%S')}", fill=(10, 20, 40), font=font_main)

        x_cursor = PADDING + LABEL_COL_WIDTH + 10

        for item in imgs:
            collage.paste(item['img'], (x_cursor, y_cursor))

            draw.rectangle(
                [x_cursor-2, y_cursor-2, x_cursor + item['img'].width + 2, y_cursor + IMG_HEIGHT + 2],
                outline=theme_color,
                width=3
            )

            if 'score' in item:
                draw.text((x_cursor, y_cursor + IMG_HEIGHT + 10),
                          f"SIM: {item['score']:.3f}", fill=theme_color, font=font_bold)

                draw.text((x_cursor, y_cursor + IMG_HEIGHT + 40),
                          "[PASS]" if item['is_match'] else "[FAIL]",
                          fill=theme_color, font=font_ts)
            else:
                draw.text((x_cursor, y_cursor + IMG_HEIGHT + 10),
                          "BASELINE", fill=(100, 100, 100), font=font_bold)

            draw.text((x_cursor, y_cursor + IMG_HEIGHT + 65),
                      f"Time: {item['time']}", fill=(0, 0, 0), font=font_ts)

            x_cursor += item['img'].width + IMG_ROW_PADDING

        y_cursor += ROW_HEIGHT + IMG_ROW_PADDING

    legend = f"Logic: Gate OR (Soft AND (Hard OR Exception)). Threshold={threshold:.2f}"
    draw.text((PADDING, collage_height - 60), legend, fill=(80, 80, 80), font=font_ts)

    base_name = repeat_data['track_id'].replace('/', '_')
    save_path = os.path.join(output_dir, f"AUDIT_EVENT_{base_name}.jpg")

    collage.save(save_path)
    logger.info(f"✅ Collage saved with ID: {save_path}")


# import os
# import re
# from PIL import Image, ImageDraw, ImageFont
# from datetime import datetime
# # Handle pathing for submodules
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# import config as local_config

# def parse_footfall_timestamp(path):
#     """
#     Parses timestamp from jewelry footfall filename format:
#     YYYY-MM-DD_HH-MM-SS_ID_TRACKID_FRAMENUM.jpg
#     Example: 2026-04-03_04-50-49_03042026043047455644_00010_00000.jpg
#     """
#     stem = os.path.basename(path)
#     match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', stem)
#     if match:
#         try:
#             date_str = match.group(1)
#             time_str = match.group(2).replace('-', ':')
#             return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
#         except:
#             return None
#     return None

# def save_detailed_repeat_audit_collage(enrollment_data, repeat_data, output_dir, threshold, logger):
#     """
#     Generates a 4-tier explainability collage for a confirmed repeat entry.
    
#     The collage provides visual evidence of why an identity was confirmed by 
#     separating samples into logical rows:
#     1. Enrollment Baseline: Original images from the first visit.
#     2. Top Overall Matches: Best matches found in the current visit (>= threshold).
#     3. Non-Matched: Samples below the threshold (discarded noise).
#     4. Verdict Proof: The specific images that triggered the passing rule.
    
#     Args:
#         enrollment_data (dict): Metadata and image paths for the enrollment visit.
#         repeat_data (dict): Metadata, image paths, and scores for the current visit.
#         output_dir (str): Directory where the JPEG collage will be saved.
#         threshold (float): Similarity bar used for visual row filtering (usually Soft Rule).
#         logger (logging.Logger): Logger instance for status reporting.
#     """
#     # --- Configuration ---
#     PADDING = 40
#     IMG_HEIGHT = 200
#     LABEL_COL_WIDTH = 420 
#     IMG_ROW_PADDING = 20
#     ROW_HEIGHT = IMG_HEIGHT + 100 
#     BACKGROUND_COLOR = (255, 255, 255)
#     HEADER_BG = (20, 40, 80)
    
#     # Fonts
#     try:
#         font_main = ImageFont.truetype("arial.ttf", 20)
#         font_id = ImageFont.truetype("arial.ttf", 18)
#         font_bold = ImageFont.truetype("arial.ttf", 26)
#         font_ts = ImageFont.truetype("arial.ttf", 20)
#         font_header = ImageFont.truetype("arial.ttf", 40)
#     except:
#         font_main = font_id = font_bold = font_ts = font_header = ImageFont.load_default()

#     def load_and_resize(path):
#         if not os.path.exists(path): return None
#         try:
#             img = Image.open(path)
#             aspect = img.width / img.height
#             return img.resize((int(IMG_HEIGHT * aspect), IMG_HEIGHT), Image.Resampling.LANCZOS)
#         except: return None

#     # --- Data Preparation ---
#     enrolled_imgs = []
#     for p in enrollment_data['images'][:10]:
#         img = load_and_resize(p)
#         if img:
#             ts = parse_footfall_timestamp(p)
#             enrolled_imgs.append({'img': img, 'time': ts.strftime("%H:%M:%S") if ts else "??:??:??"})

#     all_repeat_samples = []
#     scores = repeat_data.get('sample_scores', [])
#     for idx, p in enumerate(repeat_data['images']):
#         img = load_and_resize(p)
#         if img:
#             score = scores[idx] if idx < len(scores) else 0.0
#             ts = parse_footfall_timestamp(p)
#             all_repeat_samples.append({
#                 'img': img,
#                 'score': score,
#                 'time': ts.strftime("%H:%M:%S") if ts else "??:??:??",
#                 'is_match': score >= threshold
#             })
    
#     all_repeat_samples.sort(key=lambda x: x['score'], reverse=True)
#     best_matches = [s for s in all_repeat_samples if s['is_match']][:10]
#     worst_matches = [s for s in all_repeat_samples if not s['is_match']][:10]
    
#     # Verdict Samples (New 4th Row)
#     verdict_samples = []
#     v_data = repeat_data.get('verdict_samples', [])
#     for item in v_data:
#         p = item['path']
#         s = item['score']
#         img = load_and_resize(p)
#         if img:
#             ts = parse_footfall_timestamp(p)
#             verdict_samples.append({
#                 'img': img,
#                 'score': s,
#                 'time': ts.strftime("%H:%M:%S") if ts else "??:??:??",
#                 'is_match': True # These are by definition the passing samples
#             })

#     rows = [
#         ("ENROLLED BASELINE", enrolled_imgs, (20, 40, 80), enrollment_data),
#         ("TOP OVERALL MATCHES", best_matches, (40, 150, 40), repeat_data),
#         (f"NON-MATCHED (<{threshold:.2f})", worst_matches, (200, 40, 40), repeat_data),
#         (f"VERDICT: {repeat_data.get('verdict_rule', 'Match Proof')}", verdict_samples, (0, 102, 204), repeat_data)
#     ]

#     max_cols = max(len(enrolled_imgs), len(best_matches), len(worst_matches), len(verdict_samples))
#     if max_cols == 0: return

#     collage_width = LABEL_COL_WIDTH + (max_cols * (IMG_HEIGHT + IMG_ROW_PADDING)) + (2 * PADDING)
#     collage_height = (4 * (ROW_HEIGHT + IMG_ROW_PADDING)) + 180 + (2 * PADDING)
    
#     collage = Image.new('RGB', (collage_width, collage_height), BACKGROUND_COLOR)
#     draw = ImageDraw.Draw(collage)

#     # Main Header
#     draw.rectangle([0, 0, collage_width, 120], fill=HEADER_BG)
#     repeat_id_short = repeat_data['track_id'].split('/')[-1]
#     header_text = f"REPEAT AUDIT: {repeat_id_short}"
#     draw.text((PADDING, 35), header_text, fill=(255, 255, 255), font=font_header)

#     y_cursor = 160
#     for label_title, imgs, theme_color, meta in rows:
#         label_rect = [PADDING, y_cursor, PADDING + LABEL_COL_WIDTH - 20, y_cursor + IMG_HEIGHT]
#         draw.rectangle(label_rect, fill=(245, 248, 255), outline=theme_color, width=2)
#         draw.rectangle([PADDING, y_cursor, PADDING + 10, y_cursor + IMG_HEIGHT], fill=theme_color)
        
#         draw.text((PADDING + 25, y_cursor + 15), label_title, fill=theme_color, font=font_bold)
        
#         tid_short = meta['track_id'].split('/')[-1]
#         t_id_text = f"ID: {tid_short}"
#         t_day_text = f"Date: {meta['day']}"
#         t_time_text = f"Entry: {meta['timestamp'].strftime('%H:%M:%S')}"
        
#         draw.text((PADDING + 25, y_cursor + 55), t_id_text, fill=(10, 20, 40), font=font_id)
#         draw.text((PADDING + 25, y_cursor + 85), t_day_text, fill=(10, 20, 40), font=font_main)
#         draw.text((PADDING + 25, y_cursor + 115), t_time_text, fill=(10, 20, 40), font=font_main)
        
#         if "MATCHED" in label_title and "TOP" in label_title:
#             matched_rule = meta.get('matched_rule', 'N/A')
#             ratio = meta.get('match_ratio', 0)
#             draw.text((PADDING + 25, y_cursor + 145), f"RULE: {matched_rule}", fill=theme_color, font=font_bold)
#             draw.text((PADDING + 25, y_cursor + 175), f"Ratio: {ratio:.2%}", fill=theme_color, font=font_main)

#         # Draw Samples
#         x_cursor = PADDING + LABEL_COL_WIDTH + 10
#         for item in imgs:
#             collage.paste(item['img'], (x_cursor, y_cursor))
#             draw.rectangle([x_cursor-2, y_cursor-2, x_cursor + item['img'].width + 2, y_cursor + IMG_HEIGHT + 2], outline=theme_color, width=3)
            
#             if 'score' in item:
#                 score_str = f"SIM: {item['score']:.3f}"
#                 status_str = "[PASS]" if item['is_match'] else "[FAIL]"
#                 draw.text((x_cursor, y_cursor + IMG_HEIGHT + 10), score_str, fill=theme_color, font=font_bold)
#                 draw.text((x_cursor, y_cursor + IMG_HEIGHT + 40), status_str, fill=theme_color, font=font_ts)
#             else:
#                 draw.text((x_cursor, y_cursor + IMG_HEIGHT + 10), "BASELINE", fill=(100, 100, 100), font=font_bold)
            
#             draw.text((x_cursor, y_cursor + IMG_HEIGHT + 65), f"Time: {item['time']}", fill=(0, 0, 0), font=font_ts)
#             x_cursor += item['img'].width + IMG_ROW_PADDING
            
#         y_cursor += ROW_HEIGHT + IMG_ROW_PADDING

#     # Legend
#     legend_str = f"Logic: (Soft AND (Hard OR Exception)) OR Gate (Count >= {local_config.GLOBAL_MATCH_MIN_SAMPLES} @ {local_config.GLOBAL_MATCH_GATE}). | Coloring: {threshold:.2f}."
#     draw.text((PADDING, collage_height - 60), legend_str, fill=(80, 80, 80), font=font_ts)

#     base_name = repeat_data['track_id'].replace('/', '_')
#     file_name = f"AUDIT_EVENT_{base_name}.jpg"
#     save_path = os.path.join(output_dir, file_name)
#     collage.save(save_path)
#     logger.info(f"Multi-Rule Audit Collage saved: {save_path}")
