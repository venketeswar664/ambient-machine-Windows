import logging
import time
from datetime import datetime
from pymongo import MongoClient

import os
import sys, json

# Set the current directory to the path so that 'src' can be accessed
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..', '..')
sys.path.append(src_dir)

# Example KPI classes
from src.core.think.machine_learning.computer_vision.truck_movements import TruckMovements
from src.core.think.machine_learning.computer_vision.people_kpi import PeopleKpi
from src.core.think.machine_learning.computer_vision.unsafe_mhe_kpi import UnsafeMHEKpi
from src.utils.logger import Logger


class KPIManager:
    """
    KPIManager is responsible for calculating and storing KPI documents
    for all active cameras. It processes metadata in intervals and avoids
    reprocessing already processed metadata.
    """

    def __init__(self, db_url, db_name="sentinel", log_dir=None):
        """
        :param db_url: MongoDB connection URL
        :param db_name: Name of the database to use
        """
        self.db_client = MongoClient(db_url)
        self.db = self.db_client[db_name]
        self.active_cameras = []

        # This dictionary specifies which KPI classes to run for each camera
        self.camera_kpis_dict = {
            "D03": [TruckMovements, PeopleKpi],  # Truck and People KPI for D03
            "D04": [PeopleKpi, UnsafeMHEKpi],   # People and Unsafe MHE KPI for D04
        }

        self.logger = Logger(log_name=f'kpi_manager', log_dir=log_dir, verbose=True)

    def start_kpi_calculation(self, active_cameras, interval_sec=10, test=False):
        """
        Launch KPI calculations in an infinite loop, every `interval_sec` seconds.
        :param active_cameras: List of camera configs, each with 'device_name'
        :param interval_sec: Frequency in seconds to check and process new metadata
        """
        self.active_cameras = active_cameras

        while True:
            for camera_data in self.active_cameras:
                device_name = camera_data.get('device_name')
                if not device_name:
                    self.logger.warning("Camera data missing 'device_name'. Skipping...")
                    continue

                self.logger.info(f"Starting KPI calculations for camera: {device_name}")
                self._process_camera_kpi(device_name)

            if test:
                break

            time.sleep(interval_sec)

    def _process_camera_kpi(self, camera_name):
        """
        Processes KPIs for a specific camera based on unprocessed metadata.
        """
        # 1. Identify KPI classes
        kpi_classes = self.camera_kpis_dict.get(camera_name, [])
        if not kpi_classes:
            # self.logger.info(f"No KPI classes found for camera '{camera_name}'. Skipping.")
            return

        metadata_collection = self.db["metadata"]
        latest_time = None

        # 2. Get the latest `time_window` across all KPI collections
        for kpi_class in kpi_classes:
            kpi_collection_name = kpi_class._get_collection_name()
            kpi_collection = self.db[kpi_collection_name]

            latest_doc = kpi_collection.find_one(
                {"camera_name": camera_name},
                sort=[("time_window", -1)]
            )
            if latest_doc:
                doc_time = latest_doc["time_window"]
                if not latest_time or doc_time > latest_time:
                    latest_time = doc_time

        if latest_time:
            self.logger.info(f"Latest time_window for camera {camera_name}: {latest_time}")
        else:
            self.logger.info(f"No KPI documents found for camera: {camera_name}. Waiting for data.")
            latest_time = datetime.min  # Fresh start

        # 3. Fetch unique evidence_path values from metadata after `latest_time`
        evidence_paths = metadata_collection.distinct(
            "evidence_path", {"device_name": camera_name, "time_stamp": {"$gt": latest_time}}
        )

        if not evidence_paths:
            self.logger.info(f"No new evidence paths for camera {camera_name}. Retrying after delay.")
            time.sleep(5)  # Introduce a delay when no data is available
            return

        # Sort evidence paths and exclude the most recent one
        sorted_paths = sorted(evidence_paths)

        if len(sorted_paths) > 1:
            self.logger.info(
                f"Camera '{camera_name}' has multiple new evidence paths: {sorted_paths}"
            )

        paths_to_process = sorted_paths[:-1]  # Exclude the latest evidence_path

        if not paths_to_process:
            self.logger.info(f"No eligible evidence paths to process for camera {camera_name}.")
            return

        # 4. Process metadata for selected evidence paths
        for path in paths_to_process:
            metadata_docs = list(metadata_collection.find(
                {"device_name": camera_name, "evidence_path": path}
            ).sort("time_stamp", 1))

            if not metadata_docs:
                self.logger.warning(f"No documents found for evidence_path '{path}'. Skipping.")
                continue

            self.logger.info(
                f"Processing {len(metadata_docs)} documents for camera {camera_name}, evidence_path '{path}'."
            )
            self._calculate_and_store_kpi(camera_name, metadata_docs, evidence_path=path)

    def _calculate_and_store_kpi(self, camera_name, metadata_batch, evidence_path):
        """
        For each KPI class configured for this camera, instantiate and call its
        calculation method on the given metadata_batch. The KPI class is responsible
        for storing the results in its own collection.
        """
        if not metadata_batch:
            return

        kpi_classes = self.camera_kpis_dict.get(camera_name, [])
        if not kpi_classes:
            return

        self.logger.info(f"Calculating KPI for camera {camera_name}, evidence_path: {evidence_path}")

        for kpi_class in kpi_classes:
            try:
                kpi_instance = kpi_class()
                # Each KPI class might have a different method name; for example:
                # - `TruckMovements` might have `calculate_truck_movements(metadata, camera)`
                # - `PeopleKpi` and `UnsafeMHEKpi` might have `analyze(metadata, camera)`
                if hasattr(kpi_instance, "calculate_truck_movements"):
                    kpi_instance.calculate_truck_movements(metadata_batch, camera_name, evidence_path=evidence_path)
                elif hasattr(kpi_instance, "analyze"):
                    kpi_instance.analyze(metadata_batch, camera_name, evidence_path=evidence_path)
                else:
                    self.logger.warning(f"KPI class {kpi_class.__name__} has no recognized method to process data.")
            except Exception as e:
                self.logger.error(f"Error in {kpi_class.__name__} for camera {camera_name}: {e}")


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example active cameras
    active_cameras = [
        {"device_name": "D03"},
        {"device_name": "D04"},
    ]
    database_file = "src/configs/security/database/database-sentinel.json"
    with open(database_file) as f:
        data = json.load(f)
    db_url = data["db_url"]

    # Initialize and start KPIManage
    kpi_manager = KPIManager(db_url, db_name="sentinel", log_dir="logs/test/")
    kpi_manager.start_kpi_calculation(active_cameras, test=True)
