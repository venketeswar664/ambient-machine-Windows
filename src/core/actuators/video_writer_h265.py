import os
import cv2
import logging
import subprocess
import gc
from datetime import datetime

# class VideoWriterManager_h264_robust:
class VideoWriterManager_h265:
    """
    Robust video writing manager with fallback options for H.264 encoding issues.
    """

    def __init__(self, save_videos_flag, results_base_dir, date, frame_size, fps, logger, 
                 video_length_time=60, crf=23, preset="medium"):
        self.save_videos_flag = save_videos_flag
        self.results_base_dir = results_base_dir
        self.date = date
        self.frame_size = frame_size
        self.fps = fps
        self.logger = logger
        self.current_ffmpeg_process = None
        self.current_video_path = None
        self.video_length_time = video_length_time
        self.current_video_start_time = None
        self.crf = crf
        self.preset = preset
        self.frame_count = 0
        self.bytes_written = 0
        self.fallback_attempts = 0
        self.max_fallback_attempts = 2

    def get_ffmpeg_command(self, attempt=0):
        """Get FFmpeg command with different configurations based on attempt number."""
        
        base_command = [
            "ffmpeg",
            "-y",                           # Overwrite output file
            "-loglevel", "warning",         # Show warnings and errors
            "-f", "rawvideo",               # Input is raw video
            "-pix_fmt", "bgr24",            # OpenCV BGR format
            "-s", f"{self.frame_size[0]}x{self.frame_size[1]}",
            "-r", str(self.fps),            # Input frame rate
            "-i", "-",                      # Read from stdin
            "-c:v", "libx264",              # H.264 codec
        ]
        
        if attempt == 0:
            # First attempt: Optimized settings
            encoding_params = [
                "-preset", self.preset,
                "-tune", "zerolatency",
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p",
                "-profile:v", "main",
                "-level", "4.0",
                "-movflags", "+faststart",
                "-g", str(int(self.fps * 2)),
                "-bf", "2",
                "-refs", "2",
                "-bufsize", "2M",
                "-maxrate", "8M",
            ]
        elif attempt == 1:
            # Second attempt: More conservative settings
            encoding_params = [
                "-preset", "fast",
                "-crf", str(self.crf + 2),
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",     # Most compatible profile
                "-level", "3.1",              # Lower level
                "-movflags", "+faststart",
                "-g", str(int(self.fps)),     # Smaller GOP
                "-bufsize", "1M",
                "-maxrate", "5M",
            ]
        else:
            # Final fallback: Minimal settings
            encoding_params = [
                "-preset", "ultrafast",
                "-crf", str(self.crf + 5),
                "-pix_fmt", "yuv420p",
                # No profile specified (let FFmpeg decide)
                "-movflags", "+faststart",
            ]
        
        return base_command + encoding_params + [self.current_video_path]

    def start_new_video(self, start_time, attempt=0):
        """Start a new FFmpeg pipeline with fallback support."""
        if not self.save_videos_flag:
            return

        # Clean up any existing process first
        self.end_current_video()
        gc.collect()
        
        self.current_video_start_time = start_time
        timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Add attempt suffix if this is a retry
        suffix = f"_retry{attempt}" if attempt > 0 else ""
        video_filename = f"video_{timestamp_str}{suffix}.mp4"

        # Update directory if date changed
        if start_time.date() != self.date:
            self.results_base_dir = self.results_base_dir.replace(str(self.date), str(start_time.date()))
            self.date = start_time.date()

        self.current_video_path = os.path.join(self.results_base_dir, video_filename)
        os.makedirs(os.path.dirname(self.current_video_path), exist_ok=True)

        # Get FFmpeg command for current attempt
        ffmpeg_command = self.get_ffmpeg_command(attempt)
        
        try:
            # Start FFmpeg process
            self.current_ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * 1024 * 1024  # 10MB buffer
            )
            
            self.frame_count = 0
            self.bytes_written = 0
            self.fallback_attempts = attempt
            
            log_msg = f"Started new video: {self.current_video_path} using H.264"
            if attempt > 0:
                log_msg += f" (attempt {attempt + 1}, fallback mode)"
            self.logger.info(log_msg)
            
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg process: {e}")
            self.current_ffmpeg_process = None

    def write_frame(self, frame, current_time):
        """Write frame with error handling and automatic retry logic."""
        if not self.save_videos_flag:
            return

        if self.current_ffmpeg_process is None:
            self.start_new_video(current_time, self.fallback_attempts)

        if self.current_ffmpeg_process is not None:
            try:
                # Ensure frame size matches
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    processed_frame = cv2.resize(frame, (self.frame_size[0], self.frame_size[1]))
                else:
                    processed_frame = frame
                    
                # Ensure BGR format
                if len(processed_frame.shape) == 2:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                elif len(processed_frame.shape) == 3 and processed_frame.shape[2] == 4:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2BGR)
                
                # Write frame
                frame_bytes = processed_frame.tobytes()
                self.current_ffmpeg_process.stdin.write(frame_bytes)
                self.current_ffmpeg_process.stdin.flush()
                
                self.frame_count += 1
                self.bytes_written += len(frame_bytes)
                
                # Check for video segment completion
                elapsed_time = (current_time - self.current_video_start_time).total_seconds()
                if elapsed_time > self.video_length_time:
                    self.logger.info(f"Video segment complete: {self.frame_count} frames, "
                                   f"{self.bytes_written/1024/1024:.2f}MB written")
                    self.end_current_video()
                    self.start_new_video(current_time, 0)  # Reset attempts for new segment
                    
                # Periodic garbage collection
                if self.frame_count % 300 == 0:
                    gc.collect()
                    
            except (BrokenPipeError, OSError) as e:
                self.logger.error(f"FFmpeg pipeline error: {e}")
                
                # Try fallback if we haven't exceeded max attempts
                if self.fallback_attempts < self.max_fallback_attempts:
                    self.logger.warning(f"Attempting fallback encoding (attempt {self.fallback_attempts + 2})")
                    self.end_current_video()
                    self.start_new_video(current_time, self.fallback_attempts + 1)
                else:
                    self.logger.error("All fallback attempts failed. Stopping video recording.")
                    self.end_current_video()
                    
            except Exception as e:
                self.logger.error(f"Unexpected error writing frame: {e}")
                # For other errors, try restarting with same settings
                self.end_current_video()
                self.start_new_video(current_time, self.fallback_attempts)

    def end_current_video(self):
        """Clean up FFmpeg process."""
        if self.current_ffmpeg_process is not None:
            try:
                if self.current_ffmpeg_process.stdin:
                    self.current_ffmpeg_process.stdin.close()
                
                # Read any stderr output for debugging
                try:
                    stderr_output = self.current_ffmpeg_process.stderr.read(1024).decode()
                    if stderr_output and "error" in stderr_output.lower():
                        self.logger.warning(f"FFmpeg stderr: {stderr_output}")
                except:
                    pass
                
                # Wait for process with timeout
                try:
                    return_code = self.current_ffmpeg_process.wait(timeout=5)
                    if return_code != 0:
                        self.logger.warning(f"FFmpeg process exited with code {return_code}")
                except subprocess.TimeoutExpired:
                    self.logger.warning("FFmpeg process timeout, killing...")
                    self.current_ffmpeg_process.kill()
                    self.current_ffmpeg_process.wait()
                
                self.logger.info(f"Ended video: {self.current_video_path} ({self.frame_count} frames)")
                
            except Exception as e:
                self.logger.error(f"Error ending video: {e}")
            finally:
                self.current_ffmpeg_process = None
                gc.collect()

    def __del__(self):
        """Ensure cleanup on destruction."""
        self.end_current_video()
