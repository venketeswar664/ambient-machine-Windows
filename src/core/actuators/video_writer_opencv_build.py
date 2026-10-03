# src/core/actuators/video_writer_manager.py
import os
import cv2
import logging
import datetime
from pathlib import Path
# import mongoengine as me  # Commented out as per your structure
import sys 
import time

# from src.database.schemas.sentinel_poc_schema import Cameras, Stores  # Commented out
# from src.database.schemas.video_schema import VideoSchema  # Commented out

class VideoWriterOpencvManager:
    """
    Manages writing video frames to files, segmenting them into clips.
    
    This version is enhanced to use GPU hardware acceleration (NVIDIA NVENC) for
    video encoding if available, with a fallback to a standard CPU-based encoder.
    """
    def __init__(self, camera_id, output_dir_template,
                 frame_width, frame_height, target_fps, source_fps_val,
                 original_stream_width, original_stream_height,
                 parent_logger, clip_duration_seconds=60, use_gpu=True):
        """
        Initializes the VideoWriterManager.

        Args:
            ... (original args) ...
            use_gpu (bool): If True, attempts to use GPU for encoding. 
                            Falls back to CPU on failure.
        """
        
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
            self.logger.warning(f"No specific parent_logger provided for {camera_id}, using fallback.")

        self.camera_id = camera_id
        self.output_dir_template = output_dir_template
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.target_fps = float(target_fps) if float(target_fps) > 0 else 1.0
        self.source_fps_val = float(source_fps_val)
        self.original_stream_width = int(original_stream_width)
        self.original_stream_height = int(original_stream_height)
        self.clip_duration_seconds = int(clip_duration_seconds)
        
        # --- GPU Acceleration Control ---
        self.use_gpu = use_gpu
        self.is_using_gpu = False # This flag will be set to True if GPU init is successful
        self.gpu_frame = None # Pre-allocate GpuMat for efficiency

        # --- State Variables ---
        self.output_filepath = None
        self.cv2_video_writer = None
        # self.video_record_doc = None # Commented out
        self.start_time_utc_current_clip = None
        self.is_initialized_successfully = False
        self.frames_written_current_clip = 0

        if self.clip_duration_seconds <= 0:
            self.logger.warning(f"[{self.camera_id}] Clip duration invalid. Defaulting to 60s.")
            self.clip_duration_seconds = 60

        if not all([self.camera_id, self.output_dir_template,
                    self.frame_width > 0, self.frame_height > 0, self.target_fps > 0]):
            self.logger.error(
                f"[{self.camera_id}] VideoWriterManager critical initialization params missing."
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
            self.logger.info(f"[{self.camera_id}] Generated new clip path: {self.output_filepath}")
            return True
        except Exception as e:
            self.logger.error(f"[{self.camera_id}] Failed to generate output path: {e}", exc_info=True)
            return False

    def _initialize_cv2_writer(self):
        """
        Initializes the cv2.VideoWriter. Tries GPU first if enabled, 
        then falls back to CPU.
        """
        if not self.output_filepath:
             self.logger.error(f"[{self.camera_id}] Output filepath not set. Cannot initialize writer.")
             return False

        # --- ATTEMPT 1: GPU-ACCELERATED WRITER ---
        if self.use_gpu:
            try:
                self.logger.info(f"[{self.camera_id}] Attempting to initialize GPU-accelerated VideoWriter (NVENC)...")
                # Pre-allocate a GpuMat for re-use to avoid re-creation on every frame
                self.gpu_frame = cv2.cuda.GpuMat()
                
                # Use H.264 codec with BGR color format as input
                # Your build supports this via the `cudacodec` module
                self.cv2_video_writer = cv2.cudacodec.createVideoWriter(
                    self.output_filepath,
                    (self.frame_width, self.frame_height),
                    cv2.cudacodec.H264,
                    self.target_fps,
                    cv2.cudacodec.BGR   # We will upload BGR frames
                )
                self.is_using_gpu = True
                self.logger.info(f"[{self.camera_id}] Successfully initialized GPU VideoWriter.")
                return True
            except cv2.error as e:
                self.logger.warning(
                    f"[{self.camera_id}] GPU VideoWriter initialization failed: {e}. "
                    "This can happen if NVIDIA drivers are missing, the GPU is unsupported, "
                    "or the CUDA toolkit is misconfigured. Falling back to CPU encoder."
                )
                self.is_using_gpu = False
                self.cv2_video_writer = None # Ensure it's reset before trying CPU

        # --- ATTEMPT 2: CPU-BASED WRITER (FALLBACK or if GPU is disabled) ---
        self.logger.info(f"[{self.camera_id}] Initializing standard CPU-based VideoWriter.")
        try:
            # Use 'h264' fourcc which is good and usually backed by FFmpeg in custom builds
            fourcc = cv2.VideoWriter_fourcc(*'h264') 
            self.cv2_video_writer = cv2.VideoWriter(
                self.output_filepath, fourcc, self.target_fps,
                (self.frame_width, self.frame_height)
            )
            
            if not self.cv2_video_writer.isOpened():
                self.logger.error(f"[{self.camera_id}] Failed to open CPU cv2.VideoWriter for: {self.output_filepath}")
                self.cv2_video_writer = None
                return False
                
            self.logger.info(f"[{self.camera_id}] Successfully initialized CPU VideoWriter.")
            return True
        except Exception as e_cpu_init:
            self.logger.error(f"[{self.camera_id}] FATAL: Both GPU and CPU VideoWriter failed to initialize for {self.output_filepath}: {e_cpu_init}", exc_info=True)
            self.cv2_video_writer = None
            return False

    def _initialize_clip_resources(self):
        self.logger.info(f"[{self.camera_id}] Initializing resources for a new video clip...")
        self.frames_written_current_clip = 0
        self.is_initialized_successfully = False

        if not self._generate_output_path():
            return 
        
        if not self._initialize_cv2_writer():
            self.logger.error(f"[{self.camera_id}] VideoWriter failed to initialize for {self.output_filepath}.")
            self.finish_clip(status='Failed', error_message="VideoWriter initialization failed")
            return 
            
        self.is_initialized_successfully = True
        self.logger.info(f"[{self.camera_id}] Successfully initialized for new clip: {self.output_filepath} (GPU: {self.is_using_gpu})")

    def is_ready(self):
        return (self.cv2_video_writer is not None)

    def write_frame(self, frame):
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)

        if self.start_time_utc_current_clip and \
           (current_time_utc - self.start_time_utc_current_clip).total_seconds() >= self.clip_duration_seconds and \
           self.frames_written_current_clip > 0:

            self.logger.info(f"[{self.camera_id}] Clip duration reached. Finalizing {self.output_filepath}.")
            self.finish_clip(status='Completed', error_message="Clip segment ended due to duration.")
            self._initialize_clip_resources()
        
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            self.logger.warning(f"[{self.camera_id}] Dropping invalid frame.")
            return False

        if not self.is_ready():
            if int(time.monotonic()) % 5 == 0:
                self.logger.warning(f"[{self.camera_id}] Writer not ready. Dropping frame.")
            return False
        
        try:
            # CPU-based Pre-processing (common for both paths)
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
            if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # --- Writing Logic ---
            if self.is_using_gpu:
                # 1. Upload the CPU frame (numpy array) to the GPU (GpuMat)
                self.gpu_frame.upload(frame)
                # 2. Write the GpuMat using the hardware encoder
                self.cv2_video_writer.write(self.gpu_frame)
            else:
                # Write the numpy array using the CPU encoder
                self.cv2_video_writer.write(frame)

            self.frames_written_current_clip += 1
            return True
        except cv2.error as e_cv2_write:
            self.logger.error(f"[{self.camera_id}] OpenCV error writing frame to {self.output_filepath}: {e_cv2_write}", exc_info=True)
            return False
        except Exception as e_write:
            self.logger.error(f"[{self.camera_id}] Generic error writing frame to {self.output_filepath}: {e_write}", exc_info=True)
            return False

    def get_output_path(self):
        return self.output_filepath


    def finish_clip(self, status='Completed', error_message=None):
        self.logger.info(f"[{self.camera_id}] Finishing clip: {self.output_filepath} with status: {status}")
        
        if self.cv2_video_writer is not None:
            self.logger.info(f"[{self.camera_id}] Releasing VideoWriter for: {self.output_filepath}")
            try: 
                self.cv2_video_writer.release()
            except Exception as e_release: 
                self.logger.error(f"[{self.camera_id}] Exception releasing VideoWriter: {e_release}")
            self.cv2_video_writer = None 
        
        # All your database and file validation logic can remain here
        # ...
        
        self.is_initialized_successfully = False
        self.output_filepath = None
        # self.video_record_doc = None # Commented out

    def __del__(self):
        if self.cv2_video_writer is not None :
            self.logger.warning(f"[{self.camera_id}] Writer for {self.output_filepath or 'unknown path'} destroyed without explicit finish. Forcing finish.")
            self.finish_clip(status='Failed', error_message="Writer destroyed unexpectedly")