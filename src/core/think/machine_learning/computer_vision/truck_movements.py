import logging
from datetime import datetime
from mongoengine import Document, DateTimeField, IntField, StringField, DictField

logging.basicConfig(level=logging.INFO)

class TruckMovementCounts(Document):
    """
    MongoEngine schema for storing truck movement counts KPI.
    """
    time_window = DateTimeField(required=True)
    camera_name = StringField(required=True)
    evidence_path = StringField(required=True)

    # Storing integer counts
    shutter_open_count = IntField(default=0)
    truck_present_count = IntField(default=0)
    total_docs = IntField(default=0)

    objects_entering = IntField(default=0)
    objects_exiting = IntField(default=0)

    # New fields to store unique track count and track ID counts
    unique_track_count = IntField(default=0)  # Stores the count of unique track IDs
    track_id_counts = DictField(default={})  # Stores the occurrence count of each track ID

    meta = {
        'indexes': ['time_window']
    }


class TruckMovements:
    """
    Class for calculating truck movement KPIs from metadata.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _get_collection_name():
        """
        Return the actual collection name used by the TruckMovementCounts schema.
        """
        return TruckMovementCounts._get_collection_name()

    def lines_intersect(self, p1, p2, p3, p4):
        """
        Returns True if the line segments p1->p2 and p3->p4 intersect.
        Uses the standard orientation-based intersection test.
        """
        def orientation(a, b, c):
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if abs(val) < 1e-9:
                return 0  # collinear
            return 1 if val > 0 else 2  # 1 -> clockwise, 2 -> counterclockwise

        def on_segment(a, b, c):
            if (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1])):
                return True
            return False

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True

        return False

    def calculate_truck_movements(
        self,
        metadata_documents,
        camera_name,
        line_start=(412.59, 813.632),
        line_end=(1430, 613),
        time_window_minutes=1,
        evidence_path=None
    ):
        """
        Calculates truck movement KPIs and stores them in the database.

        Args:
            metadata_documents (list[dict]): List of metadata documents containing
                bounding box info and timestamps.
            camera_name (str): Name of the camera to set the KPI collection dynamically.
            line_start (tuple): Start point of the line for crossings.
            line_end (tuple): End point of the line for crossings.
            time_window_minutes (int): The time window for grouping movements.
        """
        self.logger.info(f"Calculating Truck Movements KPI for camera: {camera_name}...")

        if not metadata_documents:
            self.logger.warning("No metadata documents provided. Skipping KPI calculation.")
            return

        try:
            crossing_counts = {
                'objects_in': 0,
                'objects_out': 0,
                'shutter_open_count': 0,
                'truck_present_count': 0,
                'total_docs': 0,
                'unique_track_ids': set(),
                'track_id_counts': {}  # Dictionary to store track_id appearance counts
            }

            previous_positions = {}

            for doc in metadata_documents:
                timestamp = doc['time_stamp']
                crossing_counts['total_docs'] += 1

                # Check shutter open and truck presence
                if 0 in doc['labels']:
                    crossing_counts['shutter_open_count'] += 1
                if 3 in doc['labels']:
                    crossing_counts['truck_present_count'] += 1

                for tid, bbox, label in zip(doc['track_ids'], doc['bboxes'], doc['labels']):
                    # Add track_id to the set of unique IDs
                    crossing_counts['unique_track_ids'].add(tid)

                    # Increment the track_id count, converting the key to a string
                    tid_str = str(tid)  # Ensure track_id is a string
                    if tid_str in crossing_counts['track_id_counts']:
                        crossing_counts['track_id_counts'][tid_str] += 1
                    else:
                        crossing_counts['track_id_counts'][tid_str] = 1

                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = bbox[3]
                    current_pos = (center_x, center_y)

                    if tid in previous_positions:
                        prev_pos = previous_positions[tid]

                        if self.lines_intersect(prev_pos, current_pos, line_start, line_end):
                            if current_pos[1] > prev_pos[1]:
                                self.logger.info(f"Track ID {tid} moved IN with label: {label}")
                                if label in [2, 8, 11]:
                                    crossing_counts['objects_in'] += 1
                                    self.logger.info(
                                        f"Object entered. Total objects_in: {crossing_counts['objects_in']}"
                                    )
                            else:
                                self.logger.info(f"Track ID {tid} moved OUT")
                                if label in [2, 8, 11]:
                                    crossing_counts['objects_out'] += 1
                                    self.logger.info(
                                        f"Object exited. Total objects_out: {crossing_counts['objects_out']}"
                                    )

                    previous_positions[tid] = current_pos

            # Filter track IDs with counts >= 5
            crossing_counts['track_id_counts'] = {
                tid: count for tid, count in crossing_counts['track_id_counts'].items() if count >= 5
            }

            # Update the unique_track_ids set to match filtered track IDs
            crossing_counts['unique_track_ids'] = set(crossing_counts['track_id_counts'].keys())

            time_window = metadata_documents[-1]['time_stamp']

            TruckMovementCounts(
                time_window=time_window,
                camera_name=camera_name,
                shutter_open_count=crossing_counts['shutter_open_count'],
                truck_present_count=crossing_counts['truck_present_count'],
                total_docs=crossing_counts['total_docs'],
                objects_entering=crossing_counts['objects_in'],
                objects_exiting=crossing_counts['objects_out'],
                unique_track_count=len(crossing_counts['unique_track_ids']),
                track_id_counts=crossing_counts['track_id_counts'],  # Store the filtered track_id counts
                evidence_path=evidence_path
            ).save()

            self.logger.info(f"Stored TruckMovementCounts doc for time window: {time_window}")

        except Exception as e:
            self.logger.error(f"Error calculating truck movements: {e}")
