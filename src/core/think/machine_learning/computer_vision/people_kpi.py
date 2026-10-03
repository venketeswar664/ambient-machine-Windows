import logging
from datetime import datetime
from mongoengine import Document, DateTimeField, StringField, DictField, IntField, connect


###############################################################################
# 1) PeopleKpiCounts Schema
###############################################################################
class PeopleKpiCounts(Document):
    """
    MongoEngine schema for storing KPI about people not following safety
    and people staying idle within a time window.
    Each of the following fields is a DictField that maps:
        track_id (str) -> number of documents (frames) in which this was observed.
    """
    time_window = DateTimeField(required=True)
    camera_name = StringField(required=True)
    evidence_path = StringField(required=True)

    no_helmet_track_ids = DictField()   # e.g. { "track42": 3, "track99": 5 }
    no_ppe_kit_track_ids = DictField()  # e.g. { "track42": 2, "track123": 4 }
    idle_track_ids = DictField()        # e.g. { "track50": 6 }

    total_documents = IntField(default=0)  # Total number of metadata documents processed

    meta = {
        'indexes': ['time_window', 'camera_name']
    }

    @staticmethod
    def _get_collection_name():
        return "people_kpi_counts"


###############################################################################
# 2) PeopleKpi Logic
###############################################################################
class PeopleKpi:
    """
    Class for analyzing person safety compliance (helmet/ppe_kit)
    and detecting idle persons over a time window, then storing
    results in PeopleKpiCounts as dictionaries of {track_id: doc_count}.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _get_collection_name():
        return PeopleKpiCounts._get_collection_name()

    @staticmethod
    def bboxes_intersect(boxA, boxB):
        """
        Returns True if two bounding boxes (x1, y1, x2, y2) overlap.
        """
        Ax1, Ay1, Ax2, Ay2 = boxA
        Bx1, By1, Bx2, By2 = boxB
        if Ax1 > Bx2 or Bx1 > Ax2:
            return False
        if Ay1 > By2 or By1 > Ay2:
            return False
        return True

        
    def analyze(self, metadata_documents, camera_name, time_window_minutes=1, evidence_path=None):
        """
        Analyzes metadata documents (~1-minute batch) to find:
         1. People who never intersected with helmet (label=14)
         2. People who never intersected with ppe_kit (label=12)
         3. Idle people who did not move significantly in that minute.

        Stores the results, including total_documents processed.
        """
        self.logger.info(f"Analyzing PeopleKpi for camera: {camera_name}, docs: {len(metadata_documents)}")

        if not metadata_documents:
            self.logger.warning("No metadata documents, skipping PeopleKpi analysis.")
            return

        total_documents = len(metadata_documents)  # Count total metadata docs
        person_tracks = {}

        for doc in metadata_documents:
            tids = doc.get('track_ids', [])
            bboxes = doc.get('bboxes', [])
            labels = doc.get('labels', [])

            person_indices = []
            helmet_boxes = []
            ppe_boxes = []

            for i, lbl in enumerate(labels):
                if lbl in [4, 5, 6, 7, 13]:  # Person labels
                    person_indices.append(i)
                elif lbl == 14:  # helmet
                    helmet_boxes.append(bboxes[i])
                elif lbl == 12:  # ppe_kit
                    ppe_boxes.append(bboxes[i])

            for i in person_indices:
                tid = tids[i]
                person_box = bboxes[i]

                cx = (person_box[0] + person_box[2]) / 2
                cy = (person_box[1] + person_box[3]) / 2

                if tid not in person_tracks:
                    person_tracks[tid] = {
                        'positions': [],
                        'wore_helmet': False,
                        'wore_ppe': False
                    }
                person_tracks[tid]['positions'].append((cx, cy))

                # Check helmet intersection
                for hbox in helmet_boxes:
                    if self.bboxes_intersect(person_box, hbox):
                        person_tracks[tid]['wore_helmet'] = True
                        break

                # Check ppe intersection
                for pbox in ppe_boxes:
                    if self.bboxes_intersect(person_box, pbox):
                        person_tracks[tid]['wore_ppe'] = True
                        break

        no_helmet_dict = {}
        no_ppe_dict = {}
        idle_dict = {}

        idle_threshold = 20.0  # pixels

        for tid, data in person_tracks.items():
            positions = data['positions']
            doc_count_for_tid = len(positions)

            if not data['wore_helmet']:
                no_helmet_dict[str(tid)] = doc_count_for_tid

            if not data['wore_ppe']:
                no_ppe_dict[str(tid)] = doc_count_for_tid

            if doc_count_for_tid <= 1:
                idle_dict[str(tid)] = doc_count_for_tid
            else:
                all_x = [pos[0] for pos in positions]
                all_y = [pos[1] for pos in positions]
                if (max(all_x) - min(all_x)) < idle_threshold and (max(all_y) - min(all_y)) < idle_threshold:
                    idle_dict[str(tid)] = doc_count_for_tid

        time_window = metadata_documents[-1]['time_stamp']

        kpi_doc = PeopleKpiCounts(
            time_window=time_window,
            camera_name=camera_name,
            no_helmet_track_ids=no_helmet_dict,
            no_ppe_kit_track_ids=no_ppe_dict,
            idle_track_ids=idle_dict,
            total_documents=total_documents,  # Add total document count
            evidence_path = evidence_path

        )
        kpi_doc.save()

        self.logger.info(
            f"Saved PeopleKPI for {camera_name} at {time_window}. "
            f"Docs: {total_documents}, "
            f"No Helmet: {len(no_helmet_dict)}, "
            f"No PPE: {len(no_ppe_dict)}, "
            f"Idle: {len(idle_dict)}"
        )


###############################################################################
# 3) Example Usage
###############################################################################
if __name__ == "__main__":
    # Connect to MongoDB
    connect("testdb", host="localhost", port=27017)

    # Example metadata documents
    sample_docs = [
        {
            "time_stamp": datetime(2023, 12, 1, 10, 0, 0),
            "track_ids":  ["personA", "personB"],
            "bboxes":     [(100, 100, 110, 110), (200, 200, 210, 210)],
            "labels":     [4, 14]  # Person, Helmet
        },
        {
            "time_stamp": datetime(2023, 12, 1, 10, 0, 30),
            "track_ids":  ["personA", "personC"],
            "bboxes":     [(101, 101, 111, 111), (250, 250, 260, 260)],
            "labels":     [4, 12]  # Person, PPE
        },
        {
            "time_stamp": datetime(2023, 12, 1, 10, 1, 0),
            "track_ids":  ["personD"],
            "bboxes":     [(300, 300, 310, 310)],
            "labels":     [4]  # Person
        },
    ]

    # Run PeopleKpi analysis
    pkpi = PeopleKpi()
    pkpi.analyze(sample_docs, camera_name="D04", time_window_minutes=1)
