"""
A centralized service manager and scheduler.

This script reads a master YAML configuration (`services.yaml`) and manages
system cron jobs for multiple, independent services.

It operates in two modes:
1. `setup`: Clears all previously managed cron jobs and creates new ones
   based on the current `services.yaml`. This is run manually after a
   config change.
   
   Example:
   $ python src/services/service_manager.py setup

2. `run`: This is the command executed by a cron job. It runs a specific
   service script as a subprocess. It is not meant to be run manually.

   Example (as run by cron):
   $ python /path/to/repo/src/services/service_manager.py run --service video_backup_scheduler >> ...
"""

from __future__ import annotations
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from circus.watcher import Watcher
import yaml

try:
    from crontab import CronTab
except ImportError:
    CronTab = None

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "src" / "configs" / "service_manager" / "services.yaml"
LOG_DIR = ROOT / "logs" / "service_manager"
CRON_COMMENT_PREFIX = "service-manager" # Used to identify jobs managed by this script

def _setup_logging(service_name: str, level: str = "INFO") -> logging.Logger:
    """Sets up a basic logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(service_name)
    logger.setLevel(log_level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger

def load_config() -> dict:
    """Loads and validates the master services.yaml config."""
    log = logging.getLogger("service_manager")
    if not CONFIG_FILE.exists():
        log.error(f"Master config file not found at: {CONFIG_FILE}")
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "services" not in config or not isinstance(config["services"], dict):
        raise ValueError(f"Config must contain a top-level 'services' dictionary.")
    return config["services"]

import re

def _normalize_schedule(s: str) -> str:
    """Accept '@every 600s' OR a very common cron '*/N * * * *' and map to @every."""
    if not s:
        return "@every 600s"
    s = s.strip()
    if s.startswith("@every"):
        return s
    # naive: */N * * * *  -> @every Ns
    m = re.fullmatch(r"\*/(\d+)\s+\*\s+\*\s+\*\s+\*", s)
    if m:
        mins = int(m.group(1))
        return f"@every {mins*60}s"
    # fallback
    return s


def setup_jobs(logger: logging.Logger):
    """
    Reads the config and synchronizes the user's crontab.

    This function is idempotent. It first removes all jobs managed by this
    service manager before creating new ones from the config file.
    """
    if CronTab is None:
        raise RuntimeError("python-crontab is not installed. Run `pip install python-crontab`.")

    services = load_config()
    cron = CronTab(user=True)
    manager_script_path = Path(__file__).resolve()
    python_bin = sys.executable

    # 1. Clear all existing jobs managed by this script
    initial_job_count = len(cron)
    removed_count = 0
    for job in cron.find_comment(f"{CRON_COMMENT_PREFIX}:"):
        cron.remove(job)
        removed_count += 1

    if removed_count > 0:
        logger.info(f"Removed {removed_count} old service-manager job(s).")
        cron.write() # Write after removal

    created_count = 0
    for name, config in services.items():
        if not config.get("enabled", False):
            logger.info(f"Service '{name}' is disabled, skipping.")
            continue

        try:
            schedule = config["schedule"]
            script_path = ROOT / config["script"]
            args = config.get("args", [])

            if not script_path.exists():
                logger.error(f"[{name}] Script not found at '{script_path}', skipping job creation.")
                continue

            # Command that cron will run
            run_command = f'"{python_bin}" "{manager_script_path}" run --service "{name}"'

            # Setup logging for the service itself
            log_file = LOG_DIR / f"{name}.log"
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            full_command = f'{run_command} >> "{log_file}" 2>&1'
            comment = f"{CRON_COMMENT_PREFIX}:{name}"

            job = cron.new(command=full_command, comment=comment)
            job.setall(schedule)
            created_count += 1
            logger.info(f"Scheduled service '{name}' with schedule: '{schedule}'")

        except KeyError as e:
            logger.error(f"Service '{name}' is missing required key: {e}. Skipping.")
        except Exception as e:
            logger.error(f"Failed to create cron job for service '{name}': {e}")

    if created_count > 0:
        cron.write()

    logger.info(f"Setup complete. Total jobs managed: {created_count}. Current crontab size: {len(cron)}.")

def run_service(service_name: str, logger: logging.Logger):
    """
    Executes a single service defined in the configuration.
    This function is called by the cron job itself.
    """
    services = load_config()

    if service_name not in services:
        logger.error(f"Attempted to run unknown service '{service_name}'. Check your crontab and services.yaml.")
        sys.exit(1)

    config = services[service_name]
    script_path = ROOT / config["script"]
    args = config.get("args", [])

    command_to_run = [sys.executable, str(script_path)] + args

    logger.info(f"--- Running service: {service_name} ---")
    logger.info(f"Executing command: {' '.join(command_to_run)}")

    try:
        # We use subprocess.run to execute the target script and wait for it to complete.
        # stdout/stderr are already redirected by the cron command itself.
        result = subprocess.run(
            command_to_run,
            check=True,        # Raise an exception for non-zero exit codes
            cwd=ROOT,          # Run from the repository root
            text=True,         # Capture output as text (though we redirect)
            capture_output=True,
        )
        logger.info(f"Service '{service_name}' finished successfully (exit code 0).")
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"STDERR:\n{result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Service '{service_name}' failed with exit code {e.returncode}.")
        if e.stdout:
            logger.error(f"STDOUT from failed process:\n{e.stdout.strip()}")
        if e.stderr:
            logger.error(f"STDERR from failed process:\n{e.stderr.strip()}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(f"Script not found for service '{service_name}': {script_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while running '{service_name}': {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"--- Finished service: {service_name} ---")

import shlex

def _cmd_str(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("service_manager")

"""
Fixed version of build_watchers with corrected Circus stream configuration
"""

def build_watchers(attached_to_main: bool, log_dir= None) -> list[Watcher]:
    """
    Create Circus watchers from services.yaml filtered by 'attached_to_main'.

    Two shapes are supported:
      1) Dedicated runner (long-running daemon):
         runner: "src/services/main_pipeline_service.py"
         â†’ Runs continuously as a daemon

      2) Generic runner (scheduled task):
         script: "src/services/video_backup_scheduler.py"
         schedule: "@every 3600s"
         â†’ Runs periodically via generic_runner.py

    Any service missing both 'runner' and 'script' is skipped with a warning.
    """
    # Setup log directory and file handler for service_manager logs
    LOG_DIR = Path(log_dir) if log_dir else ROOT / "logs" / "service_manager"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Attach a FileHandler so the logs go to a file, not just the console
    log_file_path = LOG_DIR / "service_manager.log"
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path.resolve()) for h in LOG.handlers):
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        LOG.addHandler(file_handler)

    LOG.setLevel(logging.INFO)

    services = load_config()

    env = dict(os.environ,
               PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH','')}",
               FM_PIPELINE_CONFIG=os.environ.get("FM_PIPELINE_CONFIG", "src/configs/pipelines/main_pipeline.json"),
               FM_START_HOUR=os.environ.get("FM_START_HOUR", "10"),
               FM_STOP_HOUR=os.environ.get("FM_STOP_HOUR", "22"),
               FM_GRACE_SEC=os.environ.get("FM_GRACE_SEC", "20"),
               SERVICE_LOG_DIR=str(LOG_DIR),
               PYTHONIOENCODING="utf-8")

    watchers: list[Watcher] = []

    for name, cfg in services.items():
        if not cfg.get("enabled", False):
            continue
        if bool(cfg.get("attached_to_main", False)) != attached_to_main:
            continue

        # 1) Dedicated runner branch (long-running daemon)
        if cfg.get("runner"):
            runner_path = (ROOT / cfg["runner"]).resolve()
            if not runner_path.exists():
                LOG.warning("Service %s runner not found: %s (skipping)", name, runner_path)
                continue

            cmd = sys.executable
            args = ["-u", str(runner_path)]

            LOG.info(f"Building daemon watcher for '{name}' with runner: {runner_path}")

        # 2) Generic runner branch (scheduled task - requires 'script')
        elif cfg.get("script"):
            script_path = (ROOT / cfg["script"]).resolve()
            if not script_path.exists():
                LOG.warning("Service %s script not found: %s (skipping)", name, script_path)
                continue

            generic_runner = (ROOT / "src" / "process_supervisor" / "generic_runner.py").resolve()
            cmd = sys.executable
            args = [
                "-u",
                str(generic_runner),
                "--service", name,
                "--schedule", _normalize_schedule(cfg.get("schedule", "@every 600s")),
                "--script", str(script_path),
            ]

            if cfg.get("run_now", False):
                args.append("--run-now")

            if cfg.get("args"):
                args.append("--args")
                args.extend([str(a) for a in cfg["args"]])

            LOG.info(f"Building scheduled watcher for '{name}' via generic_runner")

        else:
            LOG.warning("Service %s has neither 'runner' nor 'script' (skipping)", name)
            continue

        # Setup log files in the dynamic log directory
        stdout_log = str((LOG_DIR / f"{name}.out.log").resolve())
        stderr_log = str((LOG_DIR / f"{name}.err.log").resolve())

        watcher_kwargs = dict(
            name=name,
            cmd=cmd,
            args=args,
            working_dir=str(ROOT),
            copy_env=True,
            env=env,
            singleton=True,
            stop_children=True,
            graceful_timeout=20,
            max_retry=5,
            retry_in=10,
            flapping_attempts=3,
            flapping_window=10,
            flapping_timeout=30,
            send_hup=False,
            close_child_stdout=False,
            close_child_stderr=False,
        )

        if os.name != 'nt':
            watcher_kwargs['stdout_stream'] = {
                'filename': stdout_log,
                'max_bytes': 20_000_000,
                'backup_count': 3,
            }
            watcher_kwargs['stderr_stream'] = {
                'filename': stderr_log,
                'max_bytes': 20_000_000,
                'backup_count': 3,
            }
        else:
            wrapper_script = (ROOT / "src" / "process_supervisor" / "win_log_wrapper.py").resolve()
            watcher_kwargs['args'] = ["-u", str(wrapper_script), stdout_log, stderr_log, watcher_kwargs['cmd']] + watcher_kwargs['args']
            watcher_kwargs['cmd'] = sys.executable

        watchers.append(Watcher(**watcher_kwargs))

        LOG.info(f"âœ“ Built watcher for '{name}': {_cmd_str([cmd] + args)}")

    if not watchers:
        LOG.warning("No watchers built for attached_to_main=%s â€” check services.yaml", attached_to_main)
    else:
        LOG.info(f"Total watchers built: {len(watchers)}")

    return watchers


def main():
    parser = argparse.ArgumentParser(description="Centralized Service Manager for cron jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # `setup` command
    parser_setup = subparsers.add_parser("setup", help="Create/update cron jobs from services.yaml.")
    parser_setup.set_defaults(func=setup_jobs)

    # `run` command
    parser_run = subparsers.add_parser("run", help="Run a specific service (for internal use by cron).")
    parser_run.add_argument("--service", required=True, help="The name of the service to run.")
    parser_run.set_defaults(func=run_service)

    args = parser.parse_args()

    logger = _setup_logging("service_manager")

    if args.command == "setup":
        args.func(logger)
    elif args.command == "run":
        args.func(args.service, logger)


if __name__ == "__main__":
    main()