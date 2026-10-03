import os
import datetime as dt
import cv2
import threading
import signal
import subprocess
from typing import Optional, Tuple
import time
import numpy as np
from src.monitoring_stack.mongodb_logger import initialize_logger



class FrameGlitchDetector:
    def __init__(self, logger=None, debug: bool = False):
        self.logger = logger
        self.debug = debug

    @staticmethod
    def _entropy_from_hist(hist: np.ndarray) -> float:
        p = hist.astype(np.float64)
        s = p.sum()
        if s <= 0:
            return 0.0
        p /= s
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def is_frame_glitched(self, frame, entropy_threshold: float = 4.0, white_threshold: float = 0.6) -> bool:
        if frame is None:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        frame_entropy = self._entropy_from_hist(hist)
        white_ratio = float(np.sum(gray > 220)) / float(gray.size)
        if self.debug:
            msg = f"entropy={frame_entropy:.2f} white_ratio={white_ratio:.2f}"
            if self.logger: self.logger.debug(msg)
            else: print(msg)
        return frame_entropy < entropy_threshold or white_ratio > white_threshold


class IP_Camera:
    """
    Connect to RTSP (GStreamer-first, H.264/H.265 auto-detect) or local file.
    Provides frame reading, optional background reader loop, and health monitoring.
    """

    def __init__(
        self,
        ip_address: str,
        device_name: str = "camera-001",
        rotation: int = 0,
        logger=None,
        zones=None,
        process_skip_frame: int = 0,
        
    ):
        self.ip_address = ip_address
        self.device_name = device_name
        self.rotation = rotation % 360
        self.logger = logger
        self.zones = zones if zones is not None else []
        self.capture = None
        self.is_video_file = False
        self.is_open = False
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.process_skip_frames = max(0, process_skip_frame)
        self._io_lock = threading.Lock()
        self._last_frame = None
        self._last_frame_time = 0.0

        self.reader_thread = None
        self.reader_stop = threading.Event()
        self.reader_running = False
        self.logger = initialize_logger(category="ip_camera")
        self.health_check_thread = None
        self.health_check_running = False
        self.is_connected = False
        self.is_corrupted = False

        self.glitch = FrameGlitchDetector(logger=self.logger, debug=False)

        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def connect(self, rtsp_codec: str = "auto"):
        if os.path.isfile(self.ip_address):
            self.is_video_file = True
            if self.logger: self.logger.info(f"Opening file: {self.ip_address}")
            self.capture = cv2.VideoCapture(self.ip_address)
        else:
            self.is_video_file = False
            if rtsp_codec == "auto":
                rtsp_codec = self._detect_stream_codec()
                if self.logger: self.logger.info(f"Detected codec: {rtsp_codec}")
            pipeline = self._gst_pipeline(self.ip_address, codec=rtsp_codec, latency_ms=200)
            if self.logger: self.logger.info(f"GStreamer connect ({rtsp_codec})")
            self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.capture.isOpened():
                alt = "h264" if rtsp_codec == "h265" else "h265"
                if self.logger: self.logger.info(f"{rtsp_codec} failed, trying {alt}")
                pipeline = self._gst_pipeline(self.ip_address, codec=alt, latency_ms=200)
                self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.capture.isOpened():
                if self.logger: self.logger.info("GStreamer failed; using FFmpeg backend")
                self.capture = cv2.VideoCapture(self.ip_address, cv2.CAP_FFMPEG)
                try:
                    self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass

        if not self.capture.isOpened():
            msg = f"Unable to connect to {self.device_name} @ {self.ip_address}"
            if self.logger: self.logger.error(msg)
            raise RuntimeError(msg)

        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Connected but failed to read initial frame")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 0.0
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0

        h, w = frame.shape[:2]
        if self.rotation in (90, 270):
            self.frame_width, self.frame_height = h, w
        else:
            self.frame_width, self.frame_height = w, h

        self.is_open = True
        self._last_frame = self._apply_rotation(frame)
        self._last_frame_time = time.time()

        try:
            backend = self.capture.getBackendName()
        except Exception:
            backend = "Unknown"
        if self.logger:
            self.logger.info(
                f"Connected to {self.device_name} via {backend} | "
                f"FPS: {self.fps:.2f} | Res: {self.frame_width}x{self.frame_height}"
            )

    def start_reader_loop(self, target_fps: Optional[float] = None):
        if self.reader_running:
            return
        self.reader_stop.clear()
        self.reader_running = True
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(target_fps,),
            daemon=True,
            name=f"{self.device_name}-reader",
        )
        self.reader_thread.start()

    def stop_reader_loop(self):
        self.reader_stop.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=5)
        self.reader_running = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.reader_running:
            frame = self.get_latest_frame(copy_frame=True)
            return (frame is not None, frame)
        if not self.capture or not self.is_open:
            return False, None
        with self._io_lock:
            ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        frame = self._apply_rotation(frame)
        self._last_frame = frame
        self._last_frame_time = time.time()
        return True, frame

    def get_latest_frame(self, copy_frame: bool = True) -> Optional[np.ndarray]:
        f = self._last_frame
        if f is None:
            return None
        return f.copy() if copy_frame else f

    def last_frame_age(self) -> float:
        if self._last_frame_time == 0.0:
            return float("inf")
        return time.time() - self._last_frame_time

    def start_health_monitor(self, interval: int = 30):
        if self.health_check_running:
            return
        self.health_check_running = True
        self.health_check_thread = threading.Thread(
            target=self._monitor_camera_health, args=(interval,),
            daemon=True, name=f"{self.device_name}-health"
        )
        self.health_check_thread.start()

    def stop_health_monitor(self):
        self.health_check_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)

    def close(self):
        self.stop_health_monitor()
        self.stop_reader_loop()
        self.is_open = False
        with self._io_lock:
            if self.capture:
                self.capture.release()
                self.capture = None
        if self.logger:
            self.logger.info(f"Camera {self.device_name} closed")

   
    def _reader_loop(self, target_fps: Optional[float]):
        interval = None
        if target_fps and target_fps > 0:
            interval = 1.0 / float(target_fps)
        while not self.reader_stop.is_set() and self.is_open:
            t0 = time.time()
            with self._io_lock:
                ok, frame = self.capture.read() if self.capture else (False, None)
            if ok and frame is not None:
                frame = self._apply_rotation(frame)
                self._last_frame = frame
                self._last_frame_time = t0
            if interval:
                spent = time.time() - t0
                to_sleep = max(0.0, interval - spent)
                if to_sleep > 0:
                    time.sleep(to_sleep)

    def _monitor_camera_health(self, interval: int):
        while self.health_check_running:
            try:
                now = time.time()
                fresh = (now - self._last_frame_time) < max(2.0, 2.0 * (1.0 / max(1.0, self.fps)) * 10.0)
                self.is_connected = bool(self.capture and self.capture.isOpened() and fresh)
                sample = self._last_frame
                self.is_corrupted = self.glitch.is_frame_glitched(sample) if sample is not None else True
                self.push_camera_status_to_db(self.is_connected, self.is_corrupted)
                if self.logger:
                    self.logger.debug(f"[{self.device_name}] connected={self.is_connected} corrupted={self.is_corrupted}")
            except Exception as e:
                if self.logger: self.logger.error(f"[{self.device_name}] Health monitor error: {e}")
            time.sleep(max(1, int(interval)))

    def _detect_stream_codec(self) -> str:
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                self.ip_address,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                codec = result.stdout.strip().lower()
                if "264" in codec: return "h264"
                if "265" in codec or "hevc" in codec: return "h265"
            return "h264"
        except Exception:
            return "h264"

    def _gst_pipeline(self, uri: str, codec: str = "h264", latency_ms: int = 200) -> str:
        c = codec.lower()
        depay = "rtph264depay" if c == "h264" else "rtph265depay"
        parse = "h264parse" if c == "h264" else "h265parse"
        dec = "avdec_h264" if c == "h264" else "avdec_h265"
        return (
            f"rtspsrc location={uri} protocols=tcp latency={latency_ms} ! "
            f"{depay} ! {parse} ! {dec} ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )

    def _apply_rotation(self, frame):
        if self.rotation == 0:
            return frame
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def push_camera_status_to_db(self, is_connected: bool, is_corrupted: bool):
        now = dt.datetime.now(dt.timezone.utc)
        try:
            if self.logger:
                self.logger.info(
                    f"[{self.device_name}] status: connected={is_connected} corrupted={is_corrupted} ts={now.isoformat()}"
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"[{self.device_name}] DB push failed: {e}")

    def _signal_handler(self, signum, _frame):
        if self.logger:
            self.logger.info(f"Signal {signum} received; shutting down")
        self.close()
