import os
import cv2
import logging
import subprocess
import gc
from datetime import datetime

class VideoWriterManager:
    """
    Manages video writing using FFmpeg with H.265 encoding through a memory-optimized real-time pipeline.
    """

    def __init__(self, save_videos_flag, results_base_dir, date, frame_size, fps, logger, 
                 video_length_time=60, crf=28, preset="ultrafast"):
        self.save_videos_flag = save_videos_flag
        self.results_base_dir = results_base_dir
        self.date = date
        self.frame_size = frame_size  # Expected format: (width, height)
        self.fps = fps
        self.logger = logger
        self.current_ffmpeg_process = None
        self.current_video_path = None
        self.video_length_time = video_length_time  # Length of each video segment in seconds
        self.current_video_start_time = None
        self.crf = crf  # Compression quality (higher = more compression, lower quality)
        self.preset = preset  # FFmpeg encoding speed preset
        self.frame_count = 0
        self.bytes_written = 0

    def start_new_video(self, start_time):
        """
        Starts a new FFmpeg pipeline with memory-optimized settings.
        """
        if not self.save_videos_flag:
            return

        # Clean up any existing process first
        self.end_current_video()
        
        # Force garbage collection before starting new video
        gc.collect()
        
        self.current_video_start_time = start_time
        timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        video_filename = f"video_{timestamp_str}.mp4"

        # Update directory if the date has changed
        if start_time.date() != self.date:
            self.results_base_dir = self.results_base_dir.replace(str(self.date), str(start_time.date()))
            self.date = start_time.date()

        self.current_video_path = os.path.join(self.results_base_dir, video_filename)
        print(f"Video path: {self.current_video_path}")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.current_video_path), exist_ok=True)

        # Memory-optimized FFmpeg command
        ffmpeg_command = [
            "ffmpeg",
            "-y",                           # Overwrite output file if it exists
            "-loglevel", "error",           # Minimal logging; only errors will be shown
            "-f", "rawvideo",               # Input is raw video
            "-pix_fmt", "bgr24",            # Pixel format (OpenCV uses BGR)
            "-s", f"{self.frame_size[0]}x{self.frame_size[1]}",  # Frame size
            "-r", str(self.fps),            # Frames per second
            "-i", "-",                      # Read input from stdin (pipe)
            "-c:v", "libx265",              # Use H.265 codec for better compression
            "-preset", self.preset,         # Encoding speed/efficiency trade-off
            "-tune", "zerolatency",         # Optimize for real-time encoding
            "-crf", str(self.crf),          # Constant Rate Factor (quality level)
            "-bufsize", "5M",               # Buffer size (keeps memory usage in check)
            "-maxrate", "5M",               # Maximum bitrate
            "-movflags", "+faststart",      # Optimize for web streaming
            "-g", str(int(self.fps * 2)),   # GOP size (keyframe interval)
            self.current_video_path
        ]

        # Start the FFmpeg process with larger pipe buffer
        self.current_ffmpeg_process = subprocess.Popen(
            ffmpeg_command, 
            stdin=subprocess.PIPE,
            bufsize=10 * 1024 * 1024  # 10MB buffer
        )
        
        self.frame_count = 0
        self.bytes_written = 0
        self.logger.info(f"Started new video: {self.current_video_path} using H.265 (memory-optimized)")

    def write_frame(self, frame, current_time):
        """
        Writes a frame to the FFmpeg process with memory optimization.
        """
        if not self.save_videos_flag:
            return

        if self.current_ffmpeg_process is None:
            self.start_new_video(current_time)

        if self.current_ffmpeg_process is not None:
            # Process the frame (creating a copy to avoid modifying the original)
            processed_frame = None
            
            try:
                # Ensure frame size matches
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    processed_frame = cv2.resize(frame, (self.frame_size[0], self.frame_size[1]))
                else:
                    processed_frame = frame  # No resize needed
                    
                # Convert grayscale to BGR if necessary
                if len(processed_frame.shape) == 2 or (len(processed_frame.shape) > 2 and processed_frame.shape[2] != 3):
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                
                # Convert frame to bytes once and reuse
                frame_bytes = processed_frame.tobytes()
                self.current_ffmpeg_process.stdin.write(frame_bytes)
                self.current_ffmpeg_process.stdin.flush()  # Ensure data is sent to the process
                
                # Track statistics
                self.frame_count += 1
                self.bytes_written += len(frame_bytes)
                
                # Only keep reference to original frame
                if processed_frame is not frame:
                    del processed_frame
                
                # Check if it's time to start a new video segment
                elapsed_time = (current_time - self.current_video_start_time).total_seconds()
                if elapsed_time > self.video_length_time:
                    self.logger.info(f"Video segment complete: {self.frame_count} frames, {self.bytes_written/1024/1024:.2f}MB written")
                    self.end_current_video()
                    self.start_new_video(current_time)
                    
                # Periodic garbage collection
                if self.frame_count % 300 == 0:  # Every 300 frames (10 seconds at 30fps)
                    gc.collect()
                    
            except BrokenPipeError:
                self.logger.error("FFmpeg pipeline closed unexpectedly")
                self.end_current_video()
                self.start_new_video(current_time)
            except Exception as e:
                self.logger.error(f"Error writing frame: {e}")
                # If there's an error, try to restart the pipeline
                self.end_current_video()
                self.start_new_video(current_time)
        else:
            self.logger.error("Failed to write frame: FFmpeg process is None")

    def end_current_video(self):
        """
        Closes the FFmpeg process cleanly with proper resource cleanup.
        """
        if self.current_ffmpeg_process is not None:
            try:
                # Close stdin pipe
                self.current_ffmpeg_process.stdin.flush()
                self.current_ffmpeg_process.stdin.close()
                
                # Wait for process to finish with timeout
                try:
                    return_code = self.current_ffmpeg_process.wait(timeout=10)
                    if return_code != 0:
                        self.logger.warning(f"FFmpeg process exited with code {return_code}")
                except subprocess.TimeoutExpired:
                    self.logger.warning("FFmpeg process did not terminate in time, killing it")
                    self.current_ffmpeg_process.kill()
                
                self.logger.info(f"Ended video: {self.current_video_path} ({self.frame_count} frames written)")
            except Exception as e:
                self.logger.error(f"Error ending video {self.current_video_path}: {e}")
            finally:
                self.current_ffmpeg_process = None
                gc.collect()  # Force garbage collection

    def __del__(self):
        """
        Ensure the FFmpeg process is closed when the object is destroyed.
        """
        self.end_current_video()