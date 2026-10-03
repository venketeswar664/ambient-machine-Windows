# src/core/think/manager/zone_manager.py

import numpy as np
from shapely.geometry import Point, Polygon, LineString, box  # added Point, box, np
from src.database.schemas.cameras_schema import Cameras
from src.database.schemas.zones_schema import Zones
# from src.database.schemas.sentinel_poc_schema import Zones, Cameras


class ZoneManager:
    """
    ZoneManager handles conversion of zone definitions (a dict of zone_name: zone_id)
    into Shapely geometries and provides utilities to compute spatial relationships.
    The ZoneManager is initialized with a single camera to only load relevant ROIs for that camera.
    """
    def __init__(self, logger, camera):
        """
        Initialize the ZoneManager with a logger and a camera.
        The camera is a dictionary containing the `device_name` of the camera.
        """
        self.logger = logger
        self.camera = camera  # Camera dictionary containing `device_name`
        self.zone_dict = {}

    @staticmethod
    def get_center(bbox):
        """
        Calculate the center of a bounding box [x1, y1, x2, y2].
        Returns a tuple (x_center, y_center).
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = bbox[3]
        return (x_center, y_center)

    def set_zones(self):
        self.zone_dict = {}
        if not self.zones:
            self.logger.warning("No zones found for this camera.")
            return

        zone_objects = Zones.objects(id__in=self.zones)
        self.zone_data = {zone.name: str(zone.id) for zone in zone_objects}
        
        for zone_name, zone_id in self.zone_data.items():
            zone_doc = Zones.objects(id=zone_id).first()
            if not zone_doc:
                self.logger.warning(f"Zone doc with ID '{zone_id}' not found.")
                continue

            vertices = zone_doc.roi  
            if not vertices:
                self.logger.warning(f"No ROI found for zone '{zone_name}'.")
                continue

            vertices_flat = [int(x) for x in vertices[0]]
            if len(vertices_flat) % 2 != 0:
                self.logger.warning(f"Odd # of coords for zone '{zone_name}'. Skipping.")
                continue

            vertices_points = [
                tuple(vertices_flat[i:i+2]) for i in range(0, len(vertices_flat), 2)
            ]

            shape = None
            shape_type = None

            if len(vertices_points) > 2:
                poly = Polygon(vertices_points)
                if poly.is_empty or not poly.is_valid:
                    self.logger.warning(f"Invalid polygon for zone '{zone_name}'. Skipping.")
                    continue
                shape = poly
                shape_type = "zone"
            elif len(vertices_points) == 2:
                line = LineString(vertices_points)
                if line.is_empty or line.length == 0:
                    self.logger.warning(f"Invalid line (zero length) for zone '{zone_name}'. Skipping.")
                    continue
                shape = line
                shape_type = "line"
            else:
                self.logger.warning(f"Insufficient points for zone '{zone_name}'. Skipping.")
                continue

            self.zone_dict[zone_name] = {
                "shape": shape,
                "type": shape_type
            }

    def check_spatial_relationship(self, shape_info, point):
        shape = shape_info["shape"]
        shape_type = shape_info["type"]

        if shape_type == "zone":
            if shape.contains(point):
                return "inside"
            elif shape.touches(point):
                return "on"
            else:
                return "outside"
            
        elif shape_type == "line":
            line_coords = list(shape.coords)
            if len(line_coords) != 2:
                return "on"
            (x1, y1), (x2, y2) = line_coords
            px, py = point.x, point.y
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) > abs(dy):
                cross_val = dx * (py - y1) - dy * (px - x1)
                if abs(cross_val) < 1e-6:
                    return "on"
                return "above" if cross_val > 0 else "below"
            else:
                cross_val = dx * (py - y1) - dy * (px - x1)
                if abs(cross_val) < 1e-6:
                    return "on"
                return "left" if cross_val > 0 else "right"
        return "unknown"

    def _calculate_iou(self, bbox, zone):
        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            self.logger.error(f"Invalid bbox format in _calculate_iou(): {bbox}")
            return 0

        try:
            x1, y1, x2, y2 = bbox[:4]
            bbox_polygon = box(x1, y1, x2, y2)
            if not isinstance(zone, Polygon) or zone.is_empty or not zone.is_valid:
                self.logger.error(f"Zone is not a valid polygon: {zone}")
                return 0
            intersection_area = bbox_polygon.intersection(zone).area
            union_area = bbox_polygon.union(zone).area
            iou_value = intersection_area / union_area if union_area > 0 else 0
            return round(iou_value, 4)
        except Exception as e:
            self.logger.error(f"Error in _calculate_iou: {e}", exc_info=True)
            return 0

    def _check_spatial_relationship_iou(self, shape_info, bbox):
        shape = shape_info["shape"]
        shape_type = shape_info["type"]
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            self.logger.warning(f"Skipping invalid bbox in _check_spatial_relationship: {bbox}")
            return 0

        if shape_type == "zone" and isinstance(shape, Polygon) and shape.is_valid:
            return self._calculate_iou(bbox, shape)

        self.logger.warning(f"Skipping IoU calculation for non-zone shape: {shape_type}")
        return 0

    def update_track_ids_status(self, track_ids_dict):
        for track_id, obj_info in track_ids_dict.items():
            center_pt = Point(ZoneManager.get_center(obj_info["bbox"]))
            bbox = obj_info["bbox"]
            instance_dict = {}
            for zone_name, zone_info in self.zone_dict.items():
                iou_value = self._check_spatial_relationship_iou(zone_info, bbox)
                location_status = self.check_spatial_relationship(zone_info, center_pt)
                if zone_name not in instance_dict:
                    instance_dict[zone_name] = {}
                instance_dict[zone_name]["iou"] = iou_value
                instance_dict[zone_name]["location"] = location_status
            obj_info["instance_dict"] = instance_dict
        return track_ids_dict
