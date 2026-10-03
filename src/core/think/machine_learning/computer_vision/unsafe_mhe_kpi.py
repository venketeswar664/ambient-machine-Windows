import logging
from datetime import datetime
from mongoengine import Document, StringField, DateTimeField, DictField, IntField


###############################################################################
# 1) UnsafeMHEKpiCounts Schema
###############################################################################
class UnsafeMHEKpiCounts(Document):
    """
    MongoEngine schema for storing KPI about unsafe MHE (Material Handling Equipment)
    movements within a time window.
    """
    time_window = DateTimeField(required=True)
    camera_name = StringField(required=True)
    evidence_path = StringField(required=True)

    # DictFields with string keys
    idle_mhe_track_ids = DictField()  # e.g. {"track42": 3, "track99": 5}
    unsafe_movement = DictField()     # e.g. {"track42_track99": 2}

    total_documents = IntField(default=0)

    meta = {
        'indexes': ['time_window', 'camera_name']
    }

    @staticmethod
    def _get_collection_name():
        return "unsafe_mhe_kpi_counts"


###############################################################################
# 2) UnsafeMHEKpi Logic
###############################################################################
class UnsafeMHEKpi:
    """
    Class for analyzing MHE safety compliance and idle movements within a time window,
    then storing results in UnsafeMHEKpiCounts as dictionaries of {track_id: doc_count}.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _get_collection_name():
        return UnsafeMHEKpiCounts._get_collection_name()

    @staticmethod
    def bboxes_close(boxA, boxB, threshold=50.0):
        """
        Returns True if the boundaries of two bounding boxes are within a given threshold.
        """
        Ax1, Ay1, Ax2, Ay2 = boxA
        Bx1, By1, Bx2, By2 = boxB

        # Calculate the horizontal and vertical distances
        horizontal_dist = max(0, max(Ax1, Bx1) - min(Ax2, Bx2))
        vertical_dist = max(0, max(Ay1, By1) - min(Ay2, By2))

        # Calculate the Euclidean distance between the closest edges
        distance = (horizontal_dist ** 2 + vertical_dist ** 2) ** 0.5
        return distance <= threshold

    @staticmethod
    def bboxes_idle(box_positions, threshold=20.0):
        """
        Returns True if all bounding boxes in the list move less than a threshold.
        """
        for i in range(1, len(box_positions)):
            Ax1, Ay1, Ax2, Ay2 = box_positions[i - 1]
            Bx1, By1, Bx2, By2 = box_positions[i]

            if (
                abs(Ax1 - Bx1) > threshold or abs(Ay1 - By1) > threshold or
                abs(Ax2 - Bx2) > threshold or abs(Ay2 - By2) > threshold
            ):
                return False
        return True

    def analyze(self, metadata_documents, camera_name, time_window_minutes=1, evidence_path=None):
        """
        Analyzes metadata documents (~1-minute batch) to find:
         1. Idle MHE (Material Handling Equipment) track IDs.
         2. Unsafe proximity between MHE track IDs.

        Stores the results, including total_documents processed.
        """
        self.logger.info(f"Analyzing UnsafeMHEKpi for camera: {camera_name}, docs: {len(metadata_documents)}")

        if not metadata_documents:
            self.logger.warning("No metadata documents, skipping UnsafeMHEKpi analysis.")
            return

        total_documents = len(metadata_documents)  # Count total metadata docs
        mhe_tracks = {}

        # Step 1: Process metadata to gather information
        for doc in metadata_documents:
            tids = doc.get('track_ids', [])
            bboxes = doc.get('bboxes', [])
            labels = doc.get('labels', [])

            mhe_indices = []
            for i, lbl in enumerate(labels):
                if lbl in [9, 10, 11]:  # MHE labels
                    mhe_indices.append(i)

            # Process each MHE track ID
            for i in mhe_indices:
                tid = tids[i]
                mhe_box = bboxes[i]

                # Add track ID if not present
                if tid not in mhe_tracks:
                    mhe_tracks[tid] = {
                        'positions': [],
                        'doc_count': 0
                    }

                # Update position and increment doc count
                mhe_tracks[tid]['positions'].append(mhe_box)
                mhe_tracks[tid]['doc_count'] += 1

        # Step 2: Identify idle MHE and unsafe proximity
        idle_mhe_dict = {}
        unsafe_movement_dict = {}
        proximity_threshold = 50.0  # Pixels for unsafe proximity

        mhe_ids = list(mhe_tracks.keys())
        for tid, data in mhe_tracks.items():
            positions = data['positions']
            doc_count_for_tid = data['doc_count']

            # Idle check using the new bboxes_idle logic
            if self.bboxes_idle(positions):
                idle_mhe_dict[str(tid)] = doc_count_for_tid

        # Unsafe proximity check with unique document-level interactions
        for i in range(len(mhe_ids)):
            for j in range(i + 1, len(mhe_ids)):
                tid1, tid2 = mhe_ids[i], mhe_ids[j]
                key = f"{tid1}_{tid2}" if tid1 < tid2 else f"{tid2}_{tid1}"
                
                interaction_detected_in_docs = set()  # Track documents where interaction is detected

                for doc in metadata_documents:
                    tids = doc.get('track_ids', [])
                    bboxes = doc.get('bboxes', [])

                    if tid1 in tids and tid2 in tids:
                        idx1 = tids.index(tid1)
                        idx2 = tids.index(tid2)

                        box1 = bboxes[idx1]
                        box2 = bboxes[idx2]

                        if self.bboxes_close(box1, box2, proximity_threshold):
                            interaction_detected_in_docs.add(str(doc['_id']))  # Unique document ID

                # Only add to unsafe_movement if there were interactions
                if interaction_detected_in_docs:
                    unsafe_movement_dict[key] = len(interaction_detected_in_docs)

        # Step 3: Save results to UnsafeMHEKpiCounts
        time_window = metadata_documents[-1]['time_stamp']

        kpi_doc = UnsafeMHEKpiCounts(
            time_window=time_window,
            camera_name=camera_name,
            evidence_path=evidence_path,
            idle_mhe_track_ids=idle_mhe_dict,
            unsafe_movement=unsafe_movement_dict,
            total_documents=total_documents
        )
        kpi_doc.save()

        self.logger.info(
            f"Saved UnsafeMHEKpi for {camera_name} at {time_window}. "
            f"Docs: {total_documents}, "
            f"Idle MHE: {len(idle_mhe_dict)}, "
            f"Unsafe Proximities: {len(unsafe_movement_dict)}"
        )
