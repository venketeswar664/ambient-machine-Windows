from __future__ import annotations
import datetime as dt, importlib, json, logging, os, signal, sys, time
import shutil
from multiprocessing import Process

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

log_dir = os.environ.get("SERVICE_LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "main_pipeline_service.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("first-machine-daemon")

def cleanup_old_frames():
    """Deletes frame directories older than 3 days."""
    base_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "results", "frames")
    if not os.path.exists(base_dir):
        return

    today = dt.date.today()
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            try:
                # Try to parse the folder name as a date (YYYY-MM-DD)
                folder_date = dt.datetime.strptime(item, "%Y-%m-%d").date()
                days_diff = (today - folder_date).days
                if days_diff >= 3:
                    log.info("Deleting old frames directory: %s (Age: %d days)", item_path, days_diff)
                    shutil.rmtree(item_path)
            except ValueError:
                # Folder is not in YYYY-MM-DD format, skip
                pass
            except Exception as e:
                log.error("Error deleting old frames directory %s: %s", item_path, e)

def _load_first_machine(pipeline_config_path: str):
    from src.core.state_manager import StateManager
    from src.monitoring_stack.mongodb_logger import initialize_logger
    with open(pipeline_config_path, "r") as f:
        pipeline_config = json.load(f)
    logger = initialize_logger(category="first_machine_service")
    sm = StateManager(pipeline_config=pipeline_config, data_dict={}, logger=logger)
    fm_name = pipeline_config.get("start_machine", "MongoMachine")
    info = pipeline_config["pipeline"][fm_name]
    module = importlib.import_module(info["module"].replace("/", ".").replace(".py", ""))
    fm_class = getattr(module, info["name"])
    return fm_class, fm_name, sm

def _run_once(pipeline_config_path: str):
    fm_class, fm_name, sm = _load_first_machine(pipeline_config_path)
    fm = fm_class(machine_name=fm_name, state_manager=sm)
    fm.execute()  # blocks; weâ€™ll terminate externally

def _now(): return dt.datetime.now()

def _seconds_until(hour: int, minute: int=0) -> int:
    now = _now()
    tgt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if tgt <= now: tgt += dt.timedelta(days=1)
    return int((tgt - now).total_seconds())

def _in_window(start_h: int, stop_h: int) -> bool:
    now = _now().time()
    return dt.time(hour=start_h) <= now < dt.time(hour=stop_h)

def _kill_tree(proc: Process, grace_s: int = 20) -> None:
    """Terminate an entire process tree so all GPU memory is released.

    Sends SIGTERM to every descendant first (orderly shutdown), waits
    `grace_s` seconds, then force-kills anything still alive with SIGKILL.
    Falls back to a plain SIGKILL on the direct child when psutil is absent.
    """
    pid = proc.pid
    if pid is None:
        return

    if _PSUTIL:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []

        # Graceful SIGTERM to all descendants
        all_procs = children + [parent]
        for p in all_procs:
            try:
                p.send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass

        # Wait for everyone to exit
        _, still_alive = psutil.wait_procs(all_procs, timeout=grace_s)

        # Force-kill survivors
        for p in still_alive:
            try:
                p.kill()
                log.warning("Force-killed lingering process pid=%s", p.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
    else:
        # Fallback: just kill the direct child
        try:
            proc.terminate()
            proc.join(timeout=grace_s)
        except Exception:
            pass
        if proc.is_alive():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    log.info("Process tree rooted at pid=%s fully stopped. CPU/memory resources released.", pid)

def _graceful_shutdown(proc: Process, drain_timeout_s: int, grace_s: int = 20) -> None:
    """
    Two-phase shutdown:
      Phase 1 â€” Send SIGTERM so the detector drains its frame queue naturally.
                 Wait up to drain_timeout_s for the process to exit on its own.
      Phase 2 â€” If it's still alive after the drain timeout, force-kill with _kill_tree.
    """
    pid = proc.pid
    if pid is None:
        return

    log.info(
        "Sending SIGTERM to pid=%s â€” waiting up to %ds for frame queue to drain...",
        pid, drain_timeout_s
    )

    # Send SIGTERM to trigger graceful shutdown_event in the detector
    try:
        if _PSUTIL:
            parent = psutil.Process(pid)
            for p in parent.children(recursive=True) + [parent]:
                try:
                    p.send_signal(signal.SIGTERM)
                except (psutil.NoSuchProcess, ProcessLookupError):
                    pass
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, Exception) as e:
        log.warning("Could not send SIGTERM to pid=%s: %s", pid, e)

    # Poll every 10s â€” exit as soon as process drains and quits naturally
    deadline = time.time() + drain_timeout_s
    while time.time() < deadline:
        if not proc.is_alive():
            log.info(
                "Process pid=%s exited cleanly after draining frame queue. CPU resources released.",
                pid
            )
            return
        remaining = int(deadline - time.time())
        log.info("Drain in progress (pid=%s)... %ds remaining before force-kill.", pid, remaining)
        time.sleep(30)

    # Drain timeout exceeded â€” force kill
    log.warning(
        "Drain timeout (%ds) exceeded for pid=%s. Force-killing now.",
        drain_timeout_s, pid
    )
    _kill_tree(proc, grace_s=grace_s)

def main():
    PIPELINE           = os.environ.get("FM_PIPELINE_CONFIG", "src/configs/pipelines/main_pipeline.json")
    START_H            = int(os.environ.get("FM_START_HOUR", "10"))
    STOP_H             = int(os.environ.get("FM_STOP_HOUR", "22"))
    GRACE_S            = int(os.environ.get("FM_GRACE_SEC", "20"))
    # How long to wait for the frame queue to drain before force-killing (default 3 hours)
    DRAIN_TIMEOUT_S    = int(os.environ.get("FM_DRAIN_TIMEOUT_HOURS", "3")) * 3600

    child: Process | None = None
    while True:
        try:
            if _in_window(START_H, STOP_H):
                if child is None or not child.is_alive():
                    log.info("Window open (%02d:00-%02d:00). Starting first machineâ€¦", START_H, STOP_H)

                    # Clean up old frames before starting the pipeline for the day
                    log.info("Checking for old frames to delete...")
                    cleanup_old_frames()

                    child = Process(target=_run_once, args=(PIPELINE,), daemon=False)
                    child.start()
                time.sleep(2)
            else:
                if child is not None and child.is_alive():
                    log.info(
                        "Window closed (%02d:00). Sending SIGTERM â€” will drain frames for up to %dh before force-kill.",
                        STOP_H, DRAIN_TIMEOUT_S // 3600
                    )
                    _graceful_shutdown(child, drain_timeout_s=DRAIN_TIMEOUT_S, grace_s=GRACE_S)
                    child = None
                secs = _seconds_until(START_H, 0)
                log.info("Sleeping %ds until next %02d:00 windowâ€¦", secs, START_H)
                step = min(secs, 60)
                slept = 0
                while slept < secs:
                    time.sleep(step)
                    slept += step
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.exception("Daemon loop error: %s", e)
            time.sleep(5)

    if child is not None and child.is_alive():
        log.info("Shutdown signal received. Draining frame queue before stopping...")
        _graceful_shutdown(child, drain_timeout_s=DRAIN_TIMEOUT_S, grace_s=GRACE_S)

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    main()
