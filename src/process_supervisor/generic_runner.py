# src/process_supervisor/generic_runner.py
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from subprocess import Popen, PIPE

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

def _make_logger(service: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    runner_log = os.path.join(log_dir, f"{service}.runner.log")
    logger = logging.getLogger(f"runner[{service}]")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(runner_log, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)
        logger.propagate = False
    return logger

def _parse_every(s: str) -> IntervalTrigger:
    # supports @every 600s / 15m / 1h
    val = s.split(None, 1)[1].strip()
    if val.endswith("s"):
        return IntervalTrigger(seconds=int(val[:-1]))
    if val.endswith("m"):
        return IntervalTrigger(minutes=int(val[:-1]))
    if val.endswith("h"):
        return IntervalTrigger(hours=int(val[:-1]))
    raise ValueError(f"Unsupported @every duration: {val}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--service", required=True)
    p.add_argument("--schedule", default=None, help="CRON '*/10 * * * *' or '@every 600s'")
    p.add_argument("--script", default=None, help="Python module/file to run each tick")
    p.add_argument("--args", nargs=argparse.REMAINDER, default=[], help="Args for the script")
    p.add_argument("--run-now", dest="run_now", action="store_true", help="Run immediately on startup")
    p.add_argument("--log-dir", default="logs/circus")
    args = p.parse_args()

    logger = _make_logger(args.service, args.log_dir)

    # IMPORTANT: must have a script
    if not args.script:
        logger.error("No --script provided. The generic runner needs a script to call each time.")
        sys.exit(2)

    script_path = args.script
    if not os.path.isabs(script_path):
        script_path = os.path.join(os.getcwd(), script_path)

    def _run_once():
        cmd = [sys.executable, "-u", script_path] + [str(a) for a in (args.args or [])]
        env = dict(
            os.environ,
            PYTHONPATH=f"{os.getcwd()}:{os.environ.get('PYTHONPATH','')}",
            PYTHONIOENCODING="utf-8",
            PYTHONUTF8="1",
        )
        logger.info(f"running: {' '.join(cmd)}")
        try:
            p = Popen(
                cmd, env=env, cwd=os.getcwd(),
                stdout=PIPE, stderr=PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
            out, err = p.communicate()
            if out:
                logger.info(f"[child stdout]\n{out.rstrip()}")
            if err:
                logger.warning(f"[child stderr]\n{err.rstrip()}")
            logger.info(f"exit code: {p.returncode}")
        except Exception as e:
            logger.exception(f"spawn failed: {e}")

    sched = BackgroundScheduler()
    if args.schedule:
        s = args.schedule.strip()
        if s.startswith("@every"):
            trig = _parse_every(s)
        else:
            trig = CronTrigger.from_crontab(s)
        sched.add_job(
            _run_once, trigger=trig, name=args.service,
            misfire_grace_time=300, coalesce=True,
        )
        sched.start()
        logger.info(f"scheduled [{args.service}] with: {args.schedule}")
        for j in sched.get_jobs():
            logger.info(f"next_run={j.next_run_time}")

    if getattr(args, "run_now", False):
        logger.info("run-now: starting initial run")
        _run_once()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("shutting down scheduler")
        try:
            sched.shutdown(wait=False)
        except Exception:
            pass

if __name__ == "__main__":
    main()
