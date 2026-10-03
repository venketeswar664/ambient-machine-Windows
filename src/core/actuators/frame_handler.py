import os
import cv2
import time
import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

class FrameHandler:
    """
    Per-camera multi-threaded frame saver.
    - No blocking on submit (detector never slows).
    - No drops: every submitted frame is scheduled.
    - Date-based paths: ./results/frames/<YYYY-MM-DD>/<device>/<type>/<timestamp>_<frame>.jpg
    - Fast JPEG encode via cv2.imencode, atomic write (.part -> os.replace).
    """

    def __init__(
        self,
        device_name: str,
        base_dir: str = "./results/frames",
        default_kind: str = "raw",           # "plotted" | "raw"
        image_format: str = "jpg",               # "jpg"|"jpeg"|"png"|"bmp"
        jpeg_quality: int = 70,
        enc_workers: int = 5,                    # CPU-bound encoders
        io_workers: int = 5,                     # IO writers
        logger=None,
        log_every_s: float = 10.0,
        high_watermark_tasks: int = 1000,        # just for logging (no blocking)
        max_width: Optional[int] = None,         # [NEW] Resize frames to max_width to reduce size
    ):
        self.device_name = device_name
        self.base_dir = base_dir
        self.default_kind = default_kind
        self.image_format = image_format.lower()
        assert self.image_format in ("jpg", "jpeg", "png", "bmp"), "Unsupported image format"
        self.jpeg_quality = int(jpeg_quality)
        self.logger = logger
        self.log_every_s = float(log_every_s)
        self.high_watermark_tasks = int(high_watermark_tasks)
        self.max_width = int(max_width) if max_width else None

        self._enc_pool = ThreadPoolExecutor(max_workers=max(1, int(enc_workers)), thread_name_prefix=f"{device_name}-enc")
        self._io_pool  = ThreadPoolExecutor(max_workers=max(1, int(io_workers)),  thread_name_prefix=f"{device_name}-io")

        self._submitted = 0
        self._written = 0
        self._failed = 0
        self._last_log = time.time()

        # precompute encode params
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality] if self.image_format in ("jpg","jpeg") else None


    def submit(
        self,
        frame,                                        # np.ndarray (H,W,C), BGR
        ts_utc: Optional[datetime.datetime],
        frame_number: int,
        kind: Optional[str] = None
    ) -> str:
        """
        Schedule a frame for encode+write. Returns intended final filepath immediately.
        Never blocks; never drops.
        """
        if ts_utc is None:
            ts_utc = datetime.datetime.now(datetime.timezone.utc)
        kind = (kind or self.default_kind)
        
        filepath = self._build_filepath(ts_utc, frame_number, kind)
        frame_copy = frame.copy()

        self._enc_pool.submit(self._encode_then_write, frame_copy, filepath)

        self._submitted += 1
        self._maybe_log_stats()
        return filepath

    def close(self):
        """Wait for all pending tasks to complete."""
        if self.logger:
            self.logger.info(f"[{self.device_name}] FrameHandlerMT closing... submitted={self._submitted}")
        self._enc_pool.shutdown(wait=True)
        self._io_pool.shutdown(wait=True)
        if self.logger:
            self.logger.info(
                f"[{self.device_name}] closed. written={self._written}, failed={self._failed}"
            )


    def _build_filepath(self, ts_utc: datetime.datetime, frame_number: int, kind: str) -> str:
        date_str = ts_utc.date().isoformat()  # YYYY-MM-DD
        ts_str = ts_utc.strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # ms precision
        root = Path(self.base_dir) / date_str / self.device_name / kind
        root.mkdir(parents=True, exist_ok=True)
        return str(root / f"{ts_str}_f{int(frame_number)}.{self.image_format}")

    def _encode_then_write(self, frame, filepath: str):
        try:
            # normalize channels
            if frame is None:
                raise ValueError("Frame is None")
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            if self.max_width and frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                new_width = self.max_width
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if self._encode_params is not None:
                ok, buf = cv2.imencode(f".{self.image_format}", frame, self._encode_params)
            else:
                ok, buf = cv2.imencode(f".{self.image_format}", frame)
            if not ok:
                raise RuntimeError("cv2.imencode failed")

            self._io_pool.submit(self._write_bytes_atomic, buf, filepath)
        except Exception as e:
            self._failed += 1
            if self.logger:
                self.logger.error(f"[{self.device_name}] encode failed for {filepath}: {e}")

    def _write_bytes_atomic(self, buf, filepath: str):
        tmp = f"{filepath}.part"
        try:
            parent = Path(filepath).parent
            parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, filepath)  # atomic on same filesystem
            self._written += 1
        except Exception as e:
            # cleanup partial
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            self._failed += 1
            if self.logger:
                self.logger.error(f"[{self.device_name}] write failed for {filepath}: {e}")

    def _maybe_log_stats(self):
        now = time.time()
        if now - self._last_log >= self.log_every_s:
            self._last_log = now
            if self.logger:
                self.logger.info(
                    f"[{self.device_name}] FrameHandlerMT stats: submitted={self._submitted}, "
                    f"written={self._written}, failed={self._failed} "
                    f"(monitor disk if submitted >> written)"
                )
            # optional alert on high backlog
            if self.high_watermark_tasks and (self._submitted - (self._written + self._failed)) > self.high_watermark_tasks:
                if self.logger:
                    self.logger.warning(
                        f"[{self.device_name}] backlog high "
                        f"({self._submitted - (self._written + self._failed)} pending). "
                        f"Disk may be the bottleneck."
                    )
