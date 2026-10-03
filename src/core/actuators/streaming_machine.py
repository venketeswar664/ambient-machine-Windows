import cv2
import numpy as np
import threading
import queue
import time
import subprocess
import sys
import signal
import torch
import logging

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


def _pick_h264_encoder():
    """Pick the first available H.264 encoder, in a sensible order."""
    for name in ("nvh264enc", "vaapih264enc", "x264enc", "avenc_h264"):
        if Gst.ElementFactory.find(name):
            return name
    return None


def _yuv_caps_for(encoder_name: str) -> str:
    """Return the YUV caps that the given encoder expects."""
    if encoder_name in ("nvh264enc", "vaapih264enc"):
        return "video/x-raw,format=NV12"
    # avenc_h264 and x264enc are happy with I420
    return "video/x-raw,format=I420"


def _pick_sink(endpoint: str, base_port: int, force_udp: bool = False):
    """Return (sink_string, is_rtsp)."""
    if not force_udp and Gst.ElementFactory.find("rtspclientsink"):
        return (
            f'rtspclientsink name=outsink location=rtsp://127.0.0.1:{base_port}/{endpoint} protocols=tcp',
            True
        )
    return ('udpsink name=outsink host=127.0.0.1 port=5000', False)



class RTSPStreamerDevice:
    """Individual device streamer that handles one camera's stream."""

    def __init__(self, device_name, resolution, fps, endpoint, base_port, webrtc_port, logger=None, use_udp_sink: bool = False):
        self.device_name = device_name
        self.resolution = resolution  # (width, height)
        self.fps = max(1, int(fps))
        self.endpoint = endpoint
        self.base_port = base_port
        self.webrtc_port = webrtc_port
        self.logger = logger or logging.getLogger(__name__)
        self.use_udp_sink = use_udp_sink

        self.frame_queue = queue.Queue(maxsize=3)
        self.running = False
        self.pipeline = None
        self.streaming_thread = None
        self.frames_sent = 0

        self.mediamtx_url = f"rtsp://localhost:{base_port}/{endpoint}"
        self.webrtc_url = f"http://localhost:{webrtc_port}/{endpoint}"

        self.blank_frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

    def _gstreamer_cmd(self):
        w, h = self.resolution
        f = self.fps
        enc = _pick_h264_encoder()
        if not enc:
            raise RuntimeError("No H.264 encoder found.")
        yuv_caps = _yuv_caps_for(enc)
        sink, is_rtsp = _pick_sink(self.endpoint, self.base_port, force_udp=self.use_udp_sink)

        # For RTSP: put the sink FIRST, then link to its request pad outsink.sink_0.
        # Let rtspclientsink create the payloader internally (no rtph264pay here).
        if is_rtsp:
            return (
                f'{sink} '  # define the named sink first
                f'appsrc name=src is-live=true format=time do-timestamp=true '
                f'caps=video/x-raw,format=BGR,width={w},height={h},framerate={f}/1 '
                f'! queue leaky=downstream max-size-buffers=10 '
                f'! videoconvert ! {yuv_caps} '
                f'! {enc} name=enc '
                f'! h264parse config-interval=1 '   
                f'! outsink.sink_0'
            )

        # UDP fallback: you MUST payload yourself, then link to udpsink normally.
        return (
            f'appsrc name=src is-live=true format=time do-timestamp=true '
            f'caps=video/x-raw,format=BGR,width={w},height={h},framerate={f}/1 '
            f'! queue leaky=downstream max-size-buffers=10 '
            f'! videoconvert ! {yuv_caps} '
            f'! {enc} name=enc '
            f'! h264parse config-interval=1 '
            f'! rtph264pay pt=96 config-interval=1 '
            f'! {sink}'
        )


    def _configure_encoder_props(self, enc):
        supported = {p.name for p in enc.list_properties()}
        def set_if_has(name, value):
            if name in supported:
                try:
                    enc.set_property(name, value)
                    self.logger.debug(f"{self.device_name}: set encoder property {name}={value}")
                except Exception as e:
                    self.logger.warning(f"{self.device_name}: failed to set {name}='{value}' ({e})")

        gop = max(self.fps, 1) * 2
        set_if_has("bitrate", 4000)
        set_if_has("gop-size", gop)
        set_if_has("key-int-max", gop)
        set_if_has("keyframe-period", gop)
        set_if_has("rc-lookahead", 0)
        set_if_has("bframes", 0)
        set_if_has("zerolatency", True)
        # FIX: Corrected the preset name from "llhp" to "low-latency-hp"
        set_if_has("preset", "low-latency-hp")
        set_if_has("tune", "zerolatency")
        set_if_has("speed-preset", "ultrafast")

    # NEW: Callback to handle GStreamer errors
    def _on_error(self, bus, message):
        err, debug_info = message.parse_error()
        self.logger.error(f"❌ GSTREAMER ERROR from element {message.src.get_name()}: {err.message}")
        self.logger.error(f"   Debug info: {debug_info if debug_info else 'none'}")
        self._restart_gstreamer()

    # NEW: Callback to handle End-Of-Stream
    def _on_eos(self, bus, message):
        self.logger.warning(f" GStreamer EOS for {self.device_name}. Restarting.")
        self._restart_gstreamer()

    def start_stream(self):
        if self.running:
            return True
        self.logger.info(f"🎥 Starting stream for {self.device_name}")

        Gst.init(None)
        self.pipeline = Gst.parse_launch(self._gstreamer_cmd())
        self.appsrc = self.pipeline.get_by_name("src")
        enc = self.pipeline.get_by_name("enc")
        self._configure_encoder_props(enc)

        # NEW: Add a message bus watch to catch errors
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)
        bus.connect("message::eos", self._on_eos)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.logger.error(f"❌ {self.device_name}: failed to set pipeline to PLAYING")
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            return False

        self.running = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()
        self.receive_frame(self.blank_frame.copy())
        
        enc_factory = enc.get_factory().get_name() if enc else "unknown"
        self.logger.info(f"📦 {self.device_name}: encoder={enc_factory}, sink={('UDP' if self.use_udp_sink else 'RTSP (mediamtx)')}")
        return True

    def _restart_gstreamer(self):
        """Restart GStreamer pipeline when it fails."""
        self.logger.warning(f"🔄 Restarting GStreamer for {self.device_name}")

        # Clean up the existing pipeline and restart it
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None

        time.sleep(1.0)
        try:
            self.start_stream()
        except Exception as e:
            self.logger.error(f"❌ {self.device_name}: restart failed: {e}")

    def _streaming_worker(self):
        """Worker thread that continuously sends frames to GStreamer."""
        self.logger.info(f"📡 Streaming worker started for {self.device_name}")
        last_frame = None
        frame_duration_ns = int(1e9 / max(self.fps, 1))
        next_pts = 0  # only used if you decide to timestamp manually

        while self.running:
            try:
                # Fetch a new frame if available
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                    last_frame = frame
                    self.frame_queue.task_done()
                except queue.Empty:
                    frame = last_frame if last_frame is not None else self.blank_frame

                # Ensure size matches the expected resolution
                if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                    frame = cv2.resize(frame, self.resolution)

                # Push to appsrc. do-timestamp=true will timestamp automatically.
                data = frame.tobytes()
                gst_buf = Gst.Buffer.new_wrapped(data)
                # If you want manual timestamps instead of do-timestamp:
                # gst_buf.pts = next_pts
                # gst_buf.duration = frame_duration_ns
                # next_pts += frame_duration_ns

                ret = self.appsrc.emit("push-buffer", gst_buf)
                if ret != Gst.FlowReturn.OK:
                    self.logger.warning(f"⚠️  {self.device_name}: push-buffer returned {ret}, restarting pipeline")
                    self._restart_gstreamer()
                    continue

                self.frames_sent += 1

                # Progress log
                if self.frames_sent % (self.fps * 10) == 0:
                    self.logger.info(f"📡 {self.device_name}: {self.frames_sent} frames streamed ✅")

            except Exception as e:
                self.logger.error(f"❌ {self.device_name}: Streaming worker error: {e}")
                time.sleep(0.5)

        self.logger.info(f"📡 Streaming worker stopped for {self.device_name}")

    def receive_frame(self, frame):
        """Receive a plotted frame for streaming with drop-oldest policy."""
        if not self.running or frame is None:
            return False

        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            dropped = 0
            while not self.frame_queue.empty() and dropped < 2:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.task_done()
                    dropped += 1
                except queue.Empty:
                    break
            try:
                self.frame_queue.put_nowait(frame)
                if dropped > 0:
                    self.logger.debug(f"{self.device_name}: Dropped {dropped} frames to make room")
                return True
            except queue.Full:
                return False

    def stop_stream(self):
        """Stop streaming and cleanup."""
        if not self.running:
            return
            
        self.logger.info(f"🛑 Stopping stream for {self.device_name}")
        self.running = False

        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)

        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        
        self.logger.info(f"✅ Stream stopped for {self.device_name} ({self.frames_sent} total frames)")

    def is_streaming(self):
        """Check if the stream is active."""
        try:
            return self.running and self.pipeline and self.pipeline.get_state(0)[1] == Gst.State.PLAYING
        except Exception:
            return False

    def get_stats(self):
        """Get streaming statistics."""
        return {
            'device_name': self.device_name,
            'is_streaming': self.is_streaming(),
            'frames_sent': self.frames_sent,
            'queue_size': self.frame_queue.qsize(),
            'webrtc_url': self.webrtc_url,
            'rtsp_url': self.mediamtx_url
        }


class RTSPStreamerManager:
    """Manager for multiple RTSP streamers - no detection, only streaming."""

    def __init__(self, base_port=8554, webrtc_port=8889, logger=None):
        self.base_port = base_port
        self.webrtc_port = webrtc_port
        self.logger = logger or logging.getLogger(__name__)
        self.streamers = {}
        self.running = True

        # Periodic statistics
        self.stats_thread = None
        self.start_statistics_thread()

        self.logger.info(f"🚀 RTSP Streamer Manager initialized")
        self.logger.info(f"📡 Base RTSP port: {base_port}")
        self.logger.info(f"🌐 WebRTC port: {webrtc_port}")

    def add_device(self, device_name, resolution=(640, 480), fps=5, endpoint=None, use_udp_sink: bool = False):
        """Add a device for streaming."""
        if device_name in self.streamers:
            self.logger.warning(f"Device {device_name} already exists")
            return False

        endpoint = endpoint or f"stream_{len(self.streamers) + 1}"

        streamer = RTSPStreamerDevice(
            device_name=device_name,
            resolution=resolution,
            fps=fps,
            endpoint=endpoint,
            base_port=self.base_port,
            webrtc_port=self.webrtc_port,
            logger=self.logger,
            use_udp_sink=use_udp_sink
        )
        self.streamers[device_name] = streamer
        self.logger.info(f"✅ Added device: {device_name} -> {endpoint} (sink={'UDP' if use_udp_sink else 'RTSP'})")
        return True

    def start_device_stream(self, device_name):
        """Start streaming for a specific device."""
        if device_name not in self.streamers:
            self.logger.error(f"Device {device_name} not found")
            return False
        return self.streamers[device_name].start_stream()

    def stop_device_stream(self, device_name):
        """Stop streaming for a specific device."""
        if device_name not in self.streamers:
            self.logger.warning(f"Device {device_name} not found")
            return False
        self.streamers[device_name].stop_stream()
        return True

    def receive_plotted_frame(self, device_name, frame):
        """Receive a plotted frame from detector for streaming."""
        if device_name not in self.streamers:
            self.logger.error(f"Device {device_name} not found in streamers")
            return False
        return self.streamers[device_name].receive_frame(frame)

    def start_all_streams(self):
        """Start all device streams."""
        success_count = 0
        for device_name in self.streamers:
            if self.start_device_stream(device_name):
                success_count += 1
        self.logger.info(f"Started {success_count}/{len(self.streamers)} streams")
        return success_count == len(self.streamers)

    def stop_all_streams(self):
        """Stop all streams and cleanup."""
        self.logger.info("🛑 Stopping all RTSP streams...")
        self.running = False
        # Stop statistics thread
        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=3.0)
        # Stop all device streams
        for device_name in list(self.streamers.keys()):
            self.stop_device_stream(device_name)
        self.streamers.clear()
        self.logger.info("✅ All RTSP streams stopped")

    def get_device_stats(self, device_name):
        """Get statistics for a specific device."""
        if device_name not in self.streamers:
            return None
        return self.streamers[device_name].get_stats()

    def get_all_stats(self):
        """Get statistics for all devices."""
        return {device_name: streamer.get_stats()
                for device_name, streamer in self.streamers.items()}

    def start_statistics_thread(self):
        """Start thread for periodic statistics logging."""
        def stats_worker():
            while self.running:
                time.sleep(30)  # Log stats every 30 seconds
                if not self.running:
                    break
                active_streams = 0
                total_frames = 0
                for _, streamer in self.streamers.items():
                    if streamer.is_streaming():
                        active_streams += 1
                        total_frames += streamer.frames_sent
                if self.streamers:
                    self.logger.info(f"📊 RTSP Stats: {active_streams}/{len(self.streamers)} active, {total_frames} total frames")
        self.stats_thread = threading.Thread(target=stats_worker, daemon=True)
        self.stats_thread.start()

    def get_webrtc_urls(self):
        """Get all WebRTC URLs for viewing streams."""
        return {name: streamer.webrtc_url for name, streamer in self.streamers.items()}

    def restart_device_stream(self, device_name):
        """Restart a specific device stream."""
        if device_name not in self.streamers:
            return False
        self.logger.info(f"🔄 Restarting stream for {device_name}")
        self.stop_device_stream(device_name)
        time.sleep(2)
        return self.start_device_stream(device_name)


# Optional: standalone test harness
def signal_handler(signum, frame):
    print("\n🛑 Shutdown signal received...")
    global rtsp_manager
    if 'rtsp_manager' in globals():
        rtsp_manager.stop_all_streams()
    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    signal.signal(signal.SIGINT, signal_handler)

    rtsp_manager = RTSPStreamerManager(base_port=8554, webrtc_port=8889, logger=logger)

    # NOTE: If you force UDP, give each device a unique UDP port in _pick_sink (or use RTSP).
    devices = [
        ("Camera-001", "stream1"),
        ("Camera-002", "stream2"),
        ("Camera-003", "stream3"),
        ("Camera-004", "stream5"),
        ("Camera-005", "stream4"),
    ]
    for device_name, endpoint in devices:
        rtsp_manager.add_device(
            device_name=device_name,
            resolution=(640, 480),
            fps=5,
            endpoint=endpoint,
            use_udp_sink=False  # prefer RTSP into mediamtx
        )

    rtsp_manager.start_all_streams()

    print("\n🌐 WebRTC Viewing URLs:")
    for device_name, url in rtsp_manager.get_webrtc_urls().items():
        print(f"   📹 {device_name}: {url}")

    print("\n✋ Press Ctrl+C to stop all streams...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all streams...")
        rtsp_manager.stop_all_streams()
