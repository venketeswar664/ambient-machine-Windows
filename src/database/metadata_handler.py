import threading
import time
import queue
import numpy as np
from src.database.schemas.metadata_schema import Metadata

class MetadataHandler:
    def __init__(self, video_path, device, logger, batch_size=500, flush_interval=5):
        self.video_path = video_path
        self.device = device
        self.logger = logger
        self.batch_size = batch_size  # Number of frames per batch
        self.flush_interval = flush_interval  # Time in seconds to force a DB flush

        self.data_queue = queue.Queue()  # Thread-safe queue
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Background thread for batch processing
        self.worker_thread = threading.Thread(target=self.batch_db_writer, daemon=True)
        self.worker_thread.start()

    def process(self, input_dict) -> dict:
        """Adds frame metadata to the queue instead of writing directly to the database."""
        try:
            self.data_queue.put(input_dict)  # Push to queue
        except Exception as e:
            self.logger.error(f"Failed to queue metadata: {e}")
        return {}

    def batch_db_writer(self):
        """Worker thread that collects and writes metadata in batches."""
        buffer = []  # Temporary buffer for batch writes
        last_flush_time = time.time()

        while not self.stop_event.is_set():
            try:
                item = None
                try:
                    item = self.data_queue.get(timeout=1)
                except queue.Empty:
                    pass

                if item:
                    buffer.append(self.format_metadata(item)) 

                if len(buffer) >= self.batch_size or (time.time() - last_flush_time) >= self.flush_interval:
                    if buffer:
                        self.flush_to_db(buffer)
                        buffer.clear()
                        last_flush_time = time.time()

            except Exception as e:
                self.logger.error(f"Error in batch_db_writer: {e}")

        if buffer:
            self.flush_to_db(buffer)

    def flush_to_db(self, buffer):
        """Flushes the buffered metadata to the database in a single batch write."""
        try:
            Metadata.objects.insert(buffer, load_bulk=False)  # Bulk insert
            self.logger.info(f"Inserted {len(buffer)} metadata records into DB.")
        except Exception as e:
            self.logger.error(f"Database batch insert error: {e}")

    def format_metadata(self, input_dict):
        """Formats metadata into MongoDB-compatible format."""
        frame_number = int(input_dict["frame_number"])
        time_stamp = input_dict["time_stamp"]
        raw_frame_path = input_dict["raw_frame_path"]
        plotted_frame_path = input_dict.get("plotted_frame_path", None)
        inference_time = float(input_dict.get("inference_time", 0.0))

        track_ids_info = input_dict.get("track_ids_info", {})
        
        if not track_ids_info:
            track_ids_info = {}  

        formatted_track_ids_info = {
            str(track_id): {
                "track_id": str(track_id),
                # "track_id_path_list": details["track_id_path_list"],
                "bbox": [int(x) for x in (details["bbox"].tolist() if isinstance(details["bbox"], np.ndarray) else details["bbox"])],
                "confidence": float(details["confidence"]),
                "label": int(details["label"]),
                "label_name": details["label_name"],
                "instance_dict": details.get("instance_dict", {})
            }
            for track_id, details in track_ids_info.items()
        }

        return Metadata(
            frame_number=frame_number,
            time_stamp=time_stamp,
            raw_frame_path=raw_frame_path,
            plotted_frame_path=plotted_frame_path,
            device=self.device,
            inference_time=inference_time,
            track_ids_info=formatted_track_ids_info,
        )

        

    def close(self):
        """Gracefully stops the worker thread and flushes remaining data."""
        self.logger.info("Stopping batch writer...")
        self.stop_event.set()
        # Wait for 5 seconds before forcing stop (to give time for the worker thread to finish)
        self.worker_thread.join(timeout=5)
        if self.worker_thread.is_alive():
            self.logger.warning("Batch writer thread did not finish in time.")
        else:
            self.logger.info("Batch writer stopped.")
