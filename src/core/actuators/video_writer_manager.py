# src/core/actuators/video_writer_manager.py
import os
import cv2
import logging
import datetime
from pathlib import Path
import mongoengine as me
import sys 
import time # Import time for logging frequency control

class VideoWriterManager:
    def __init__(self, camera_id, output_dir_template,
                 frame_width, frame_height, target_fps, source_fps_val,
                 original_stream_width, original_stream_height,
                 parent_logger, clip_duration_seconds=60):
        
        if parent_logger:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"VideoWriterManager.{camera_id}.fallback")
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            self.logger.warning(f"No specific parent_logger provided to VideoWriterManager for {camera_id}, using basic fallback.")

        self.camera_id = camera_id

        self.output_dir_template = output_dir_template

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.target_fps = float(target_fps) if float(target_fps) > 0 else 1.0
        self.source_fps_val = float(source_fps_val)

        self.original_stream_width = int(original_stream_width)
        self.original_stream_height = int(original_stream_height)
        
        self.clip_duration_seconds = int(clip_duration_seconds)
        if self.clip_duration_seconds <=0:
            self.logger.warning(f"[{self.camera_id}] Clip duration ({self.clip_duration_seconds}s) is invalid. Defaulting to 60s.")
            self.clip_duration_seconds = 60

        self.output_filepath = None
        self.cv2_video_writer = None
        self.video_record_doc = None
        self.start_time_utc_current_clip = None
        self.is_initialized_successfully = False
        self.frames_written_current_clip = 0

        if not all([self.camera_id, self.output_dir_template,
                    self.frame_width > 0, self.frame_height > 0, self.target_fps > 0,
                    self.original_stream_width > 0, self.original_stream_height > 0,
                    self.clip_duration_seconds > 0]):
            self.logger.error(
                f"[{self.camera_id}] VideoWriterManager critical initialization params missing or invalid."
            )
            return 

        self._initialize_clip_resources()

    def _generate_output_path(self):
        self.start_time_utc_current_clip = datetime.datetime.now(datetime.timezone.utc)
        timestamp_str = self.start_time_utc_current_clip.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"clip_{timestamp_str}.mp4"
        try:
            clip_specific_dir = Path(self.output_dir_template)
            clip_specific_dir.mkdir(parents=True, exist_ok=True)
            self.output_filepath = str(clip_specific_dir / filename)
            self.logger.info(f"[{self.camera_id}] Generated output filepath for new clip: {self.output_filepath}")
            return True
        except Exception as e:
            self.logger.error(f"[{self.camera_id}] Error generating output path structure for clip: {e}", exc_info=True)
            fallback_dir = Path(self.output_dir_template).parent / "path_generation_errors"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self.output_filepath = str(fallback_dir / f"ERROR_{self.camera_id}_{timestamp_str}.mp4")
            self.logger.critical(f"[{self.camera_id}] Set output_filepath to an error path: {self.output_filepath}")
            return False

    def _initialize_cv2_writer(self):
        if not self.output_filepath:
             self.logger.error(f"[{self.camera_id}] Output filepath not set. Cannot initialize OpenCV writer.")
             return False
        try:
            Path(self.output_filepath).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.cv2_video_writer = cv2.VideoWriter(
                self.output_filepath, fourcc, self.target_fps,
                (self.frame_width, self.frame_height)
            )

            self.frame_number = 0
            if not self.cv2_video_writer.isOpened():
                self.logger.error(f"[{self.camera_id}] Failed to open cv2.VideoWriter for new clip: {self.output_filepath}")
                self.cv2_video_writer = None
                return False
            self.logger.info(f"[{self.camera_id}] cv2.VideoWriter opened successfully for new clip: {self.output_filepath}")
            return True
        except Exception as e_cv2_init:
            self.logger.error(f"[{self.camera_id}] Exception during cv2.VideoWriter init for {self.output_filepath}: {e_cv2_init}", exc_info=True)
            self.cv2_video_writer = None
            return False

    def _initialize_clip_resources(self):
        self.logger.info(f"[{self.camera_id}] Initializing resources for a new video clip segment...")
        self.frames_written_current_clip = 0
        self.is_initialized_successfully = False

        if not self._generate_output_path():
            self.logger.error(f"[{self.camera_id}] Failed to generate output path for new clip. Cannot initialize.")
            return 
        
        
        if not self._initialize_cv2_writer():
            self.logger.error(f"[{self.camera_id}] OpenCV VideoWriter failed to initialize for {self.output_filepath}.")
            # Use 'Failed' status and provide error message
            self.finish_clip(status='Failed', error_message="OpenCV VideoWriter initialization failed")
            return 
            
        self.is_initialized_successfully = True
        self.logger.info(f"[{self.camera_id}] Successfully initialized resources for new clip segment: {self.output_filepath}")


    def is_ready(self):
        return (self.cv2_video_writer is not None and 
                self.cv2_video_writer.isOpened())


    def write_frame(self, frame):
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)

        if self.start_time_utc_current_clip and \
           (current_time_utc - self.start_time_utc_current_clip).total_seconds() >= self.clip_duration_seconds and \
           self.frames_written_current_clip > 0:

            self.logger.info(f"[{self.camera_id}] Clip duration ({self.clip_duration_seconds}s) reached for {self.output_filepath}. "
                             f"Finalizing current clip with {self.frames_written_current_clip} frames.")
            # Use 'Completed' status and provide context in error_message (which becomes processing_summary)
            self.finish_clip(status='Completed', error_message="Clip segment ended due to duration.")

            self.logger.info(f"[{self.camera_id}] Initializing resources for next clip segment...")
            self._initialize_clip_resources()
        
                # 1) Reject None or empty arrays immediately
        if frame is None:
            self.logger.warning(f"[{self.camera_id}] Dropping frame: frame is None")
            return False
        if not hasattr(frame, "shape") or frame.size == 0:
            self.logger.warning(f"[{self.camera_id}] Dropping frame: invalid shape or zero size ({getattr(frame, 'shape', None)})")
            return False

        if not self.is_ready():
            if int(time.monotonic()) % 5 == 0:
                self.logger.warning(f"[{self.camera_id}] VideoWriterManager not ready. Cannot write frame. Current path: {self.output_filepath}")
            return False

        
        
        try:
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                # self.logger.debug(f"[{self.camera_id}] Resizing frame from {frame.shape[1]}x{frame.shape[0]} to {self.frame_width}x{self.frame_height}.")
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            self.cv2_video_writer.write(frame)
            self.frame_number +=1
            self.frames_written_current_clip += 1
            return True
        except cv2.error as e_cv2_write:
            self.logger.error(f"[{self.camera_id}] OpenCV error writing frame to {self.output_filepath}: {e_cv2_write}", exc_info=True)
            return False
        except Exception as e_write:
            self.logger.error(f"[{self.camera_id}] Generic error writing frame to {self.output_filepath}: {e_write}", exc_info=True)
            return False

    def finish_clip(self, status='Completed', error_message=None):

        self.logger.info(f"[{self.camera_id}] Finishing clip: {self.output_filepath} with proposed status: {status}")
        end_time_utc = datetime.datetime.now(datetime.timezone.utc)
        current_output_filepath = self.output_filepath 

        if self.cv2_video_writer is not None:
            if self.cv2_video_writer.isOpened():
                self.logger.info(f"[{self.camera_id}] Releasing cv2.VideoWriter for: {current_output_filepath}")
                try: self.cv2_video_writer.release()
                except Exception as e_release: self.logger.error(f"[{self.camera_id}] Exception releasing VideoWriter: {e_release}")
            self.cv2_video_writer = None 
        
        self.is_initialized_successfully = False
        self.output_filepath = None
        self.video_record_doc = None

    def get_output_path(self):
        return self.output_filepath

    def get_video_record_id(self):
        return str(self.video_record_doc.id) if self.video_record_doc and hasattr(self.video_record_doc, 'id') and self.video_record_doc.id else None

    def __del__(self):
        if self.cv2_video_writer is not None :
            self.logger.warning(f"[{self.camera_id}] VideoWriterManager for {self.output_filepath if self.output_filepath else 'unknown path'} being destroyed but seems active. Forcing finish_clip.")
            # Use 'Failed' status and provide context in error_message
            self.finish_clip(status='Failed', error_message="Writer destroyed before explicit finish_clip call (via __del__)")
        elif self.cv2_video_writer is not None and self.cv2_video_writer.isOpened():
             self.logger.warning(f"[{self.camera_id}] Releasing cv2.VideoWriter for {self.output_filepath if self.output_filepath else 'unknown path'} in __del__.")
             try: self.cv2_video_writer.release()
             except Exception as e_del_release: self.logger.error(f"[{self.camera_id}] Error releasing in __del__: {e_del_release}")