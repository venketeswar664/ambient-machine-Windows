#!/usr/bin/env python3
"""
scheduled_runner.py

Two modes of operation:

1) If you supply BOTH --start-time and --end-time on the command line:
   - The script will IGNORE last_run.txt entirely.
   - It will convert your input times (assumed Asia/Kolkata local) to UTC,
     split that interval into 10-minute chunks, process each chunk once,
     and then exit. It does NOT write to last_run.txt in this mode.

2) If you supply NO arguments (i.e. run “as is”):
   - The script opens/locks last_run.txt, reads the previous UTC timestamp,
   - Computes “now UTC,” slices [last_run → now] into 10-minute chunks,
   - Processes chunk by chunk, writing each chunk’s end back to last_run.txt,
     so next time it picks up where it left off. Locking ensures no overlap.

After every call to process_metadata_for_duration(...), if the local‐time hour of
chunk_start is 23 (11 PM) or any full hour between 00 and 11 (i.e. midnight → 11 AM),
and if minute == 00, we invoke bin_and_archive_tracks.parse_input_times(...) for the
preceding full hour.
"""

import os
import sys
import argparse
import datetime
import dotenv
import mongoengine as me
import fcntl
import logging
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import threading

# ────────── Import parse_input_times from your archiving script ─────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
try:
    from push_selected_track_ids import parse_input_times  # noqa: E402
except ImportError as e:
    print(f"ERROR: Could not import parse_input_times: {e}")
    sys.exit(1)

# ────────── Logging Setup ──────────────────────────────────────────────────────

_now = datetime.datetime.now()
_date_str = _now.strftime("%Y-%m-%d")
_time_str = _now.strftime("%H%M%S")

LOG_DIR = Path(__file__).resolve().parent / "logs" / "track_ids_creation" / _date_str / "create_track_ids"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = LOG_DIR / f"scheduled_runner_{_time_str}.logs"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# ─────────────── Configuration ──────────────────LAS
LAST_RUN_DIR = Path(__file__).resolve().parent / "last_run"

LAST_RUN_FILE = LAST_RUN_DIR / "create_track_ids_last_run.txt"
LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)

CHUNK_MINUTES = 10

# Import your existing pipeline function
try:
    from create_track_id import process_metadata_for_duration  # noqa: E402
except ImportError as e:
    logger.error(f"Could not import process_metadata_for_duration: {e}")
    sys.exit(1)

LOCAL_TZ = ZoneInfo("Asia/Kolkata")


# ─────────────── Helper Functions ──────────────────

def parse_iso_to_utc(iso_str: str) -> datetime.datetime:
    """
    Interpret iso_str (e.g. "2025-06-01T10:00:00") as Asia/Kolkata local time
    (if no timezone is present), then convert to UTC. Returns a timezone-aware UTC datetime.
    """
    dt = datetime.datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(datetime.timezone.utc)


def chunk_intervals(start: datetime.datetime, end: datetime.datetime, chunk_minutes: int):
    """
    Yield consecutive (chunk_start, chunk_end) pairs of length chunk_minutes
    between start (inclusive) and end (exclusive/equal). Both must be UTC-aware.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=datetime.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=datetime.timezone.utc)

    current = start
    delta = datetime.timedelta(minutes=chunk_minutes)
    while current < end:
        nxt = current + delta
        if nxt > end:
            nxt = end
        yield (current, nxt)
        current = nxt


def connect_to_mongo():
    """Connect to MongoDB using MONGODB_URI from .env or environment."""
    dotenv.load_dotenv()
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    parsed = urlparse(mongo_uri)
    db_name = parsed.path.lstrip("/") or "mydatabase"
    try:
        me.connect(db=db_name, host=mongo_uri, UuidRepresentation='standard')
        logger.info(f"Connected to MongoDB '{db_name}'.")
    except Exception as exc:
        logger.error(f"ERROR connecting to MongoDB: {exc}")
        sys.exit(1)


def disconnect_from_mongo():
    """Disconnect from MongoDB."""
    try:
        me.disconnect()
    except Exception as exc:
        logger.error(f"ERROR disconnecting MongoDB: {exc}")


def acquire_lock_and_open_last_run() -> "IO[Any]":
    """
    Open LAST_RUN_FILE in read/write (“a+”) mode (creating it if needed),
    then acquire an exclusive non-blocking flock. If lock already held,
    log a message and exit(0). Otherwise return the open file handle.
    """
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LAST_RUN_FILE, mode="a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("Another instance is still running. Exiting.")
        fh.close()
        sys.exit(0)
    return fh


def _today_10am_utc() -> datetime.datetime:
    """
    Compute today's 10:00 AM in Asia/Kolkata and convert it to UTC.
    """
    today_local = datetime.datetime.now(LOCAL_TZ).date()
    ten_am_local = datetime.datetime(
        year=today_local.year,
        month=today_local.month,
        day=today_local.day,
        hour=9,
        minute=0,
        second=0,
        tzinfo=LOCAL_TZ
    )
    return ten_am_local.astimezone(datetime.timezone.utc)


def read_last_run_time_from_file(fh) -> datetime.datetime:
    """
    Given an open file handle (already flock-locked), read its first line
    as an ISO-8601 timestamp. If empty, return today's 10 AM (local) → UTC.
    If parsing succeeds but is before today, also return today 10 AM → UTC.
    Otherwise, return the parsed UTC datetime.
    """
    fh.seek(0)
    content = fh.readline().strip()

    if not content:
        return _today_10am_utc()

    try:
        dt = datetime.datetime.fromisoformat(content)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)

        dt_local = dt.astimezone(LOCAL_TZ)
        today_local = datetime.datetime.now(LOCAL_TZ).date()
        local_time = datetime.datetime.now(LOCAL_TZ).time()
        if dt_local.date() < today_local and (local_time > datetime.time(8, 0)):
            return _today_10am_utc()
        return dt

    except Exception:
        logger.warning(f"Could not parse '{content}'. Using today 10 AM local → UTC.")
        return _today_10am_utc()


def write_last_run_time_to_file(fh, dt_utc: datetime.datetime) -> None:
    """
    Given the locked file handle, overwrite the file with dt_utc.isoformat() on line 1.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(datetime.timezone.utc)

    txt = dt_utc.isoformat()
    fh.seek(0)
    fh.truncate()
    fh.write(txt + "\n")
    fh.flush()


# ─────────────── Main ──────────────────

def main():
    parser = argparse.ArgumentParser(
        description="scheduled_runner.py: "
                    "By default, uses last_run.txt + locking + 10-min chunking.  "
                    "If you supply BOTH --start-time and --end-time, "
                    "it runs that window once (10-min chunks) and exits."
    )
    parser.add_argument(
        "--start-time",
        help="(Optional) ISO-8601 local (Asia/Kolkata) start, e.g. '2025-06-01T10:00:00'."
    )
    parser.add_argument(
        "--end-time",
        help="(Optional) ISO-8601 local (Asia/Kolkata) end, e.g. '2025-06-01T22:00:00'."
    )

    create_track_ids = True

    args = parser.parse_args()

    # ─── Mode 1: One-off interval (if both --start-time and --end-time are given) ───────────────
    if args.start_time and args.end_time:
        try:
            start_utc = parse_iso_to_utc(args.start_time)
            end_utc = parse_iso_to_utc(args.end_time)
        except Exception as e:
            logger.error(f"ERROR parsing times: {e}")
            sys.exit(1)

        if end_utc <= start_utc:
            logger.error("--end-time must be strictly after --start-time.")
            sys.exit(1)

        logger.info(
            f"Running ONE-OFF from {start_utc.isoformat()} → {end_utc.isoformat()} (UTC)"
        )

        connect_to_mongo()
        try:
            intervals = list(chunk_intervals(
                start=start_utc, end=end_utc, chunk_minutes=CHUNK_MINUTES
            ))
            logger.info(
                f"Splitting into {len(intervals)} chunk(s) of up to {CHUNK_MINUTES} minutes each."
            )
            MODEL_NAME = os.getenv("MODEL_NAME", "")

            for chunk_start, chunk_end in intervals:
                logger.info(
                    f"Processing chunk: {chunk_start.isoformat()} → {chunk_end.isoformat()}"
                )

                try:

                    if create_track_ids:
                        # 1) Run your existing pipeline
                        process_metadata_for_duration(
                            start_dt_utc=chunk_start,
                            end_dt_utc=chunk_end,
                            model_name_arg=MODEL_NAME,
                            logger=logger
                        )
                        logger.info(
                            f"SUCCESS for chunk {chunk_start.isoformat()} → {chunk_end.isoformat()}"
                        )

                    else:

                        # 2) After pipeline is done, check if local hour is 23 or [0..11] and minute == 00
                        local_chunk_start = chunk_end.astimezone(LOCAL_TZ)
                        hour_local   = local_chunk_start.hour
                        minute_local = local_chunk_start.minute

                        if minute_local == 0 and (11 <= hour_local <= 23):
                            end_naive = local_chunk_start.replace(tzinfo=None)
                            start_naive = (local_chunk_start - datetime.timedelta(hours=1)) \
                                        .replace(tzinfo=None)

                            start_iso_local = start_naive.isoformat()
                            end_iso_local   = end_naive.isoformat()

                            logger.info(
                                f"→ Local time {hour_local:02d}:{minute_local:02d} detected."
                                f" Invoking parse_input_times for preceding hour "
                                f"{start_iso_local} → {end_iso_local}."
                            )
                            try:
                                parse_input_times(
                                    start_iso_local,
                                    end_iso_local,
                                    args,
                                    logger
                                )
                            except Exception as exc:
                                logger.error(f"parse_input_times raised an exception: {exc}")

                except Exception as exc:
                    logger.error(
                        f"ERROR in chunk {chunk_start.isoformat()} → {chunk_end.isoformat()}: {exc}"
                    )
                    break

        finally:
            disconnect_from_mongo()

        logger.info("ONE-OFF run complete. Exiting.")
        sys.exit(0)

    # ─── Mode 2: “normal” scheduled mode (no explicit times provided) ───────────────
    fh = acquire_lock_and_open_last_run()
    try:
        connect_to_mongo()

        last_run = read_last_run_time_from_file(fh)
        now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

        if now_utc <= last_run:
            logger.info("No new data (now ≤ last_run). Exiting.")
            return

        logger.info(
            f"Running from last_run={last_run.isoformat()} → now={now_utc.isoformat()} (UTC)"
        )

        intervals = list(chunk_intervals(
            start=last_run, end=now_utc, chunk_minutes=CHUNK_MINUTES
        ))
        logger.info(
            f"Splitting into {len(intervals)} chunk(s) of up to {CHUNK_MINUTES} minutes each."
        )
        MODEL_NAME = os.getenv("MODEL_NAME", "")

        for chunk_start, chunk_end in intervals:
            logger.info(
                f"Processing chunk: {chunk_start.isoformat()} → {chunk_end.isoformat()}"
            )
            try:
                # 1) Run your existing pipeline
                process_metadata_for_duration(
                    start_dt_utc=chunk_start,
                    end_dt_utc=chunk_end,
                    model_name_arg=MODEL_NAME,
                    logger=logger
                )
                write_last_run_time_to_file(fh, chunk_end)
                logger.info(f"SUCCESS → updated last_run to {chunk_end.isoformat()}")

                # # 2) After pipeline is done, check if local hour is 23 or [0..11] and minute == 00
                # local_chunk_start = chunk_end.astimezone(LOCAL_TZ)
                # hour_local   = local_chunk_start.hour
                # minute_local = local_chunk_start.minute

                # if minute_local == 0 and (11 <= hour_local <= 23):
                #     end_naive = local_chunk_start.replace(tzinfo=None)
                #     start_naive = (local_chunk_start - datetime.timedelta(hours=1)) \
                #                   .replace(tzinfo=None)

                #     start_iso_local = start_naive.isoformat()
                #     end_iso_local   = end_naive.isoformat()

                #     logger.info(
                #         f"→ Local time {hour_local:02d}:{minute_local:02d} detected."
                #         f" Invoking parse_input_times for preceding hour "
                #         f"{start_iso_local} → {end_iso_local}."
                #     )

                #     try:
                #         parse_input_times(
                #             start_iso_local,
                #             end_iso_local,
                #             args,
                #             logger
                #         )
                #     except Exception as exc:
                #         logger.error(f"parse_input_times raised an exception: {exc}")





                    # try:

                    #     def parse_input_times_thread():
                    #         try:
                    #             parse_input_times(
                    #                 start_iso_local,
                    #                 end_iso_local,
                    #                 args,
                    #                 logger
                    #             )
                    #         except Exception as exc:
                    #             logger.error(f"parse_input_times raised an exception: {exc}")

                    #     parse_thread = threading.Thread(target=parse_input_times_thread)
                    #     parse_thread.start()
                    # except Exception as exc:
                    #     logger.error(f"Thread parse_input_times raised an exception: {exc}")

            except Exception as exc:
                logger.error(
                    f"ERROR in chunk {chunk_start.isoformat()} → {chunk_end.isoformat()}: {exc}"
                )
                break

    finally:
        disconnect_from_mongo()
        try:
            fh.close()
        except Exception:
            pass
        logger.info("Exiting; lock released.")


if __name__ == "__main__":
    main()

# python src/scripts/scheduled_runner.py --start-time 2025-06-29T10:00:00 --end-time 2025-06-29T22:00:00


    # parser.add_argument(
    #     "--start-time",
    #     help="(Optional) ISO-8601 local (Asia/Kolkata) start, e.g. '2025-06-01T10:00:00'."
    # )
    # parser.add_argument(
    #     "--end-time",
    #     help="(Optional) ISO-8601 local (Asia/Kolkata) end, e.g. '2025-06-01T22:00:00'."
    # )