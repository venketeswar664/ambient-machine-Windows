# -*- coding: utf-8 -*-
'''
Created on 29-May-2024 03:24
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''

import cv2
import numpy as np
from shapely.geometry import Polygon, LineString
# import seaborn as sns


def plot_dict(frame, info_dict, starting_point='top_right'):
    # Define padding and initial position
    padding = 10
    
    # Define font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)  # White color
    font_thickness = 1
    line_type = cv2.LINE_AA
    
    # Calculate the maximum width of the labels text box
    max_label_width = 0
    text_height = 0
    for label, count in info_dict.items():
        text_size = cv2.getTextSize(f'{label}: {count}', font, font_scale, font_thickness)[0]
        max_label_width = max(max_label_width, text_size[0])
        text_height = text_size[1]

    # Calculate the background rectangle dimensions
    background_height = (len(info_dict) + 1) * (text_height + padding)
    background_width = max_label_width + 2 * padding

    # Set the starting position based on the specified starting point
    if starting_point == 'top_right':
        x_start = frame.shape[1] - padding
        y_start = padding
        top_left_corner = (x_start - background_width, y_start)
        bottom_right_corner = (x_start, y_start + background_height)
    elif starting_point == 'top_left':
        x_start = padding
        y_start = padding
        top_left_corner = (x_start, y_start)
        bottom_right_corner = (x_start + background_width, y_start + background_height)
    elif starting_point == 'bottom_right':
        x_start = frame.shape[1] - padding
        y_start = frame.shape[0] - padding - background_height
        top_left_corner = (x_start - background_width, y_start)
        bottom_right_corner = (x_start, y_start + background_height)
    elif starting_point == 'bottom_left':
        x_start = padding
        y_start = frame.shape[0] - padding - background_height
        top_left_corner = (x_start, y_start)
        bottom_right_corner = (x_start + background_width, y_start + background_height)
    else:
        raise ValueError("Invalid starting point. Choose from 'top_right', 'top_left', 'bottom_right', 'bottom_left'.")

    # Draw background rectangle
    cv2.rectangle(frame, top_left_corner, bottom_right_corner, (146, 90, 16), -1)  # Black background

    # Draw each label text
    y_position = y_start + text_height + padding
    for label, count in info_dict.items():
        text = f'{label}: {count}'
        cv2.putText(frame, text, (top_left_corner[0] + padding, y_position), font, font_scale, font_color, font_thickness, line_type)
        y_position += text_height + padding

    return frame

import cv2
import numpy as np
from shapely.geometry import Polygon, LineString

def plot_shapes(frame, shapes, color_polygon=(139, 42, 242), color_line=(139, 42, 242), thickness=2, center_text=True, alpha=0.6):
    """
    Plot multiple polygons and lines on the given frame.
    """
    # Ensure frame has 3 channels
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for shape_name, shape_info in shapes.items():
        if not isinstance(shape_info, dict) or "shape" not in shape_info:
            continue  # Skip invalid entries

        shape = shape_info["shape"]
        shape_type = shape_info.get("type", "unknown")

        if isinstance(shape, Polygon):
            if shape.exterior is None:
                print(f"Skipping {shape_name}, invalid Polygon.")
                continue

            x, y = shape.exterior.xy
            polygon_vertices = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))

            # print(f"Drawing Polygon: {shape_name}, Points: {polygon_vertices}")

            # Draw the polygon outline
            overlay = frame.copy()
            cv2.polylines(overlay, [polygon_vertices], isClosed=True, color=color_polygon, thickness=thickness)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw text at the centroid
            if center_text:
                centroid = shape.centroid
                text_size = cv2.getTextSize(shape_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_position = (int(centroid.x - text_size[0] / 2), int(centroid.y + text_size[1] / 2))

                overlay = frame.copy()
                cv2.putText(overlay, shape_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color_polygon, thickness=2)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        elif isinstance(shape, LineString):
            x, y = shape.xy
            line_vertices = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))

            # print(f"Drawing Line: {shape_name}, Points: {line_vertices}")

            overlay = frame.copy()
            cv2.polylines(overlay, [line_vertices], isClosed=False, color=color_line, thickness=thickness)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw text at midpoint
            if center_text:
                mid_x = int(np.mean(x))
                mid_y = int(np.mean(y))

                text_size = cv2.getTextSize(shape_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_position = (mid_x - text_size[0] // 2, mid_y + text_size[1] // 2)

                overlay = frame.copy()
                cv2.putText(overlay, shape_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color_line, thickness=2)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        else:
            print(f"Skipping {shape_name}, unsupported shape type: {type(shape)}")

    return frame



def _get_center(bbox):
    """
    Calculate the center of the bounding box.
    bbox is in the format [x1, y1, x2, y2].
    """
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = bbox[3]
    return (x_center, y_center)

def plot_point_and_trackid(plotted_frame, boxes, track_ids, labels):
    """
    Plot points and track IDs on the given frame based on bounding boxes.

    Args:
    - plotted_frame: numpy array representing the input image frame.
    - boxes: list of bounding boxes, each box given as [x1, y1, x2, y2].
    - track_ids: list of integers, the track IDs corresponding to each bounding box.

    Returns:
    - plotted_frame: numpy array representing the frame with points and track IDs plotted.
    """
    # palette = sns.color_palette(None, 3)


    for box, track_id, label in zip(boxes, track_ids, labels):
        # Calculate the center of the bounding box
        (x_center, y_center) = _get_center(box)

        # Define the color and size of the circle and text
        emp_color = (231,184,48)  # blue
        cst_color = (68,204,102)  # green
        radius = 3
        thickness = -1  # Fill the circle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (255, 255, 255)
        background_color = (30, 30, 30) # subtle dark
        

        if label == 0:
            color = cst_color
        elif label == 1:
            color = emp_color

        else:
            color = (235,45,58)
        
        # color = palette[label]
        # # Draw the center point
        # print("x_center, y_center", x_center, y_center)
        # print("plotted_frame", plotted_frame)

        cv2.circle(plotted_frame, (int(x_center), int(y_center)), radius, color, thickness)

        # Put the track ID near the point
        offset = 10
        text_position = (int(x_center) - offset, int(y_center) - offset)
        cv2.putText(plotted_frame, str(track_id), text_position, font, font_scale, color, font_thickness)

    return plotted_frame

def plot_faces_with_labels(frame, bbox, label, confidence):
    """
    Plot bounding boxes and labels for detected faces on the frame.

    Args:
    - frame: numpy array representing the input image frame.
    - bboxes: list of bounding boxes, each box given as [x1, y1, x2, y2].
    - labels: list of labels corresponding to each bounding box.
    - confidences: list of confidence scores corresponding to each bounding box.

    Returns:
    - frame: numpy array representing the frame with faces and labels plotted.
    """
 
    x1, y1, x2, y2 = bbox
    text = f"{label}"
    print(text)
    color = (0, 255, 0)  # Green color for bounding box

    # Draw bounding box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Draw label and confidence
    cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame