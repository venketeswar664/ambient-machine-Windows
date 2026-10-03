# src/services/camera_config_reconciler.py
import os, sys, time, json, logging
from typing import Dict, Set, Tuple
from bson import ObjectId
from pymongo import MongoClient
from circus.client import CircusClient

LOG = logging.getLogger("camera_config_reconciler")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://seawoods:sentinelMongo%40123@localhost:27017/sentinel?authSource=sentinel&replicaSet=rs0&directConnection=true")
MONGO_DB  = os.environ.get("MONGODB_DBNAME", "sentinel")
CIRCUS_ENDPOINT = os.environ.get("CIRCUS_ENDPOINT", "tcp://127.0.0.1:5565")
PROJECT_ROOT = os.getcwd()
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "circus")
os.makedirs(LOG_DIR, exist_ok=True)

# Service type to script mapping
SERVICE_CONFIGS = {
    "dock safety": {
        "script": "truck_safety.py",
        "prefix": "dock_safety_"
    },
    "hdr safety": {
        "script": "hdr_safety.py",
        "prefix": "hdr_safety_"
    },
    "entry service":{
        "script": "entry_service.py",
        "prefix": "entry_service_"
    },
    "age service": {
        "script": "age_service.py",
        "prefix": "age_service_"   
    },
    "gender service": {
        "script": "gender_service.py",
        "prefix": "gender_service_"   
    },
}

def _p(msg, *args):
    s = msg % args if args else msg
    print(f"[reconciler] {s}", flush=True)
    LOG.info(s)

def _send(client: CircusClient, command: str, properties: dict | None = None) -> dict:
    payload = {"command": command}
    if properties:
        payload["properties"] = properties
    resp = client.call(payload)
    _p("circus call %s props=%s -> %s",
       command,
       json.dumps(properties or {}, default=str),
       json.dumps(resp, default=str))
    return resp

def _send_with_retry(client: CircusClient, command: str, properties: dict | None = None, retries: int = 10) -> dict:
    for attempt in range(retries):
        resp = _send(client, command, properties)
        if resp.get("status") == "ok":
            return resp
        reason = str(resp.get("reason", "")).lower()
        if "arbiter is already running" in reason or "already running" in reason:
            _p("WARN: command %s blocked by arbiter (%s), retry %d/%d in 1s...", command, reason, attempt + 1, retries)
            time.sleep(1)
            continue    
        return resp
    return resp

def safety_watcher_name(camera_oid: ObjectId, service_type: str) -> str:
    """Generate watcher name based on service type and camera ID"""
    config = SERVICE_CONFIGS.get(service_type)
    if not config:
        return f"unknown_safety_{str(camera_oid)}"
    return f"{config['prefix']}{str(camera_oid)}"

def list_running_watchers(client: CircusClient) -> Set[str]:
    try:
        resp = _send(client, "list")
        if resp.get("status") == "ok":
            return set(resp.get("watchers", []))
        _p("ERROR: circus list failed: %s", resp)
    except Exception as e:
        _p("EXCEPTION: circus list error: %s", e)
    return set()

def _add_watcher_any(client: CircusClient, properties: dict) -> dict:
    # Try 'add' (current)
    resp = _send_with_retry(client, "add", properties)
    if resp.get("status") == "ok":
        return resp
    _p("WARN: 'add' failed (%s), trying legacy 'add_watcher'…", resp.get("reason"))
    # Legacy (older circus)
    return _send_with_retry(client, "add_watcher", properties)

# def add_safety_watcher(client: CircusClient, camera_oid: ObjectId, model_path: str, service_type: str):
#     """Add a watcher for a specific safety service type"""
#     config = SERVICE_CONFIGS.get(service_type)
#     if not config:
#         _p("ERROR: Unknown service type '%s'", service_type)
#         return
    
#     name = safety_watcher_name(camera_oid, service_type)
#     script = os.path.join(PROJECT_ROOT, "src", "services", config["script"])
    
#     if not os.path.exists(script):
#         _p("ERROR: Script not found: %s", script)
#         return
    
#     args_list = [sys.executable, "-u", script, "--camera-id", str(camera_oid), "--model-path", model_path]

#     env = dict(os.environ)
#     env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH','')}"

#     # Streams need a 'class' key in most circus versions
#     stdout_stream = {
#         "class": "FileStream",
#         "filename": os.path.join(LOG_DIR, f"{name}.out.log"),
#         "max_bytes": 20_000_000,
#         "backup_count": 3,
#     }
#     stderr_stream = {
#         "class": "FileStream",
#         "filename": os.path.join(LOG_DIR, f"{name}.err.log"),
#         "max_bytes": 20_000_000,
#         "backup_count": 3,
#     }

#     props_add = {
#         "name": name,
#         "cmd": args_list[0],
#         "args": " ".join(args_list[1:]),
#         "working_dir": PROJECT_ROOT,
#         "copy_env": True,
#         "env": env,
#         "singleton": True,
#         "stop_children": True,
#         "graceful_timeout": 20,
#         "max_retry": 5,
#         "retry_in": 10,
#         "stdout_stream": stdout_stream,
#         "stderr_stream": stderr_stream,
#     }

#     _p("Adding watcher %s (type=%s) model_path=%s", name, service_type, model_path)
#     resp = _add_watcher_any(client, props_add)
#     if resp.get("status") != "ok":
#         _p("ERROR: add watcher failed for %s: %s", name, resp)
#         return

#     # start
#     resp = _send(client, "start", {"name": name})
#     if resp.get("status") != "ok":
#         _p("ERROR: start watcher failed for %s: %s", name, resp)
#     else:
#         _p("Started watcher %s", name)


def add_safety_watcher(client: CircusClient, camera_oid: ObjectId, model_path: str, service_type: str):
    """
    Add a watcher for a specific safety or entry service type.
    Automatically handles scheduled vs. continuous scripts.
    """
    config = SERVICE_CONFIGS.get(service_type)
    if not config:
        _p("ERROR: Unknown service type '%s'", service_type)
        return

    name = safety_watcher_name(camera_oid, service_type)
    script = os.path.join(PROJECT_ROOT, "src", "services", config["script"])
    if not os.path.exists(script):
        _p("ERROR: Script not found: %s", script)
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env["PYTHONIOENCODING"] = "utf-8"

    # Mongo URI (you can centralize this in your environment or settings file)
    MONGO_URI = "mongodb://sentinel:sentinelMongo%40123@localhost:27017/sentinel_test?directConnection=true"

    # Build the command dynamically depending on the service type
    if service_type == "entry service":
        # Use generic_runner for scheduling @every 60s
        generic_runner = os.path.join(PROJECT_ROOT, "src", "process_supervisor", "generic_runner.py")
        args_list = [
            sys.executable, "-u", generic_runner,
            "--service", name,
            "--schedule", "@every 60s",
            "--script", script,
            "--args",
            "--camera-id", str(camera_oid),
            "--mongo-uri", MONGO_URI
        ]
    else:
        # Continuous services (dock safety, hdr safety, etc.)
        args_list = [
            sys.executable, "-u", script,
            "--camera-id", str(camera_oid),
            "--model-path", model_path
        ]

    stdout_log = os.path.join(LOG_DIR, f"{name}.out.log")
    stderr_log = os.path.join(LOG_DIR, f"{name}.err.log")

    props_add = {
        "name": name,
        "cmd": args_list[0],
        "args": " ".join(args_list[1:]),
        "working_dir": PROJECT_ROOT,
        "copy_env": True,
        "env": env,
        "singleton": True,
        "stop_children": True,
        "graceful_timeout": 20,
        "max_retry": 5,
        "retry_in": 10,
    }

    if os.name != 'nt':
        props_add["stdout_stream"] = {
            "class": "FileStream",
            "filename": stdout_log,
            "max_bytes": 20_000_000,
            "backup_count": 3,
        }
        props_add["stderr_stream"] = {
            "class": "FileStream",
            "filename": stderr_log,
            "max_bytes": 20_000_000,
            "backup_count": 3,
        }
    else:
        import shlex
        wrapper_script = os.path.join(PROJECT_ROOT, "src", "process_supervisor", "win_log_wrapper.py")
        orig_cmd = props_add["cmd"]
        orig_args = args_list[1:]
        
        props_add["cmd"] = sys.executable
        new_args = ["-u", wrapper_script, stdout_log, stderr_log, orig_cmd] + orig_args
        props_add["args"] = " ".join(shlex.quote(str(a)) for a in new_args)

    _p("Adding watcher %s (type=%s)", name, service_type)
    resp = _add_watcher_any(client, props_add)
    if resp.get("status") != "ok":
        _p("ERROR: add watcher failed for %s: %s", name, resp)
        return

    resp = _send_with_retry(client, "start", {"name": name})
    if resp.get("status") != "ok":
        _p("ERROR: start watcher failed for %s: %s", name, resp)
    else:
        _p("✅ Started watcher %s for %s", name, service_type)

def stop_and_rm_watcher(client: CircusClient, name: str):
    try:
        _send_with_retry(client, "stop", {"name": name})
    except Exception:
        pass
    try:
        _send_with_retry(client, "rm", {"name": name})
        _p("Removed watcher %s", name)
    except Exception as e:
        _p("ERROR removing %s: %s", name, e)

def service_doc(services_coll, sid: ObjectId) -> dict | None:
    return services_coll.find_one({"_id": sid}, {"name": 1, "keywords": 1, "model": 1})

def get_service_type(sdoc: dict | None) -> str | None:
    """Determine the service type from service document"""
    if not sdoc:
        return None
    
    name = (sdoc.get("name") or "").lower().strip()
    
    # Check for exact matches or keyword matches
    for service_type in SERVICE_CONFIGS.keys():
        if service_type in name:
            return service_type
    
    # Check keywords
    kws = [k.lower().strip() for k in (sdoc.get("keywords") or [])]
    for keyword in kws:
        for service_type in SERVICE_CONFIGS.keys():
            if service_type in keyword:
                return service_type
    
    return None

def model_path_from_service(models_coll, sdoc: dict) -> str | None:
    mid = sdoc.get("model")
    if not mid:
        return None
    mdoc = models_coll.find_one({"_id": mid}, {"modelPath": 1})
    return mdoc.get("modelPath") if mdoc else None

def get_store_camera_oids(db) -> set:
    """
    Returns the set of camera ObjectIds that belong to this machine's store.
    Uses STORE_ID env var. If not set, returns None (meaning: all cameras allowed).
    """
    # HARDCODED for testing — replace with os.environ.get("STORE_ID") for multi-machine deployment
    store_id_str = "6a3d7f02cc1c4c00c19dd6f4"
    # store_id_str = os.environ.get("STORE_ID")

    if not store_id_str:
        return None  # No filter — process all stores

    try:
        store_oid = ObjectId(store_id_str)
    except Exception:
        _p("WARN: Invalid STORE_ID '%s' — processing all cameras", store_id_str)
        return None

    # Fetch all active cameras belonging to this store
    camera_oids = set(
        doc["_id"]
        for doc in db["cameras"].find(
            {"store": store_oid, "active": True},
            {"_id": 1}
        )
    )
    _p("Store filter active: store=%s, matched %d camera(s)", store_id_str, len(camera_oids))
    return camera_oids

def compute_desired(db) -> Dict[str, dict]:
    """
    Compute desired watchers from cameraConfig, filtered to this machine's store.
    Returns dict mapping watcher_name -> {camera_oid, model_path, service_type}
    """
    desired = {}
    services_coll = db["services"]
    models_coll   = db["models"]

    # Get the set of camera IDs allowed for this store (None = no filter)
    allowed_camera_oids = get_store_camera_oids(db)

    cursor = db["cameraConfig"].find({}, {"cameraOid": 1, "camera_oid": 1, "services": 1})
    for cc in cursor:
        camera_oid = cc.get("cameraOid") or cc.get("camera_oid")
        if not camera_oid:
            continue

        # Skip cameras that don't belong to this machine's store
        if allowed_camera_oids is not None and camera_oid not in allowed_camera_oids:
            _p("Skipping cameraConfig camera=%s (not in store)", camera_oid)
            continue

        _p("Evaluating cameraConfig camera=%s", camera_oid)
        for item in (cc.get("services") or []):
            sid = item.get("service")
            if not sid:
                continue
            
            sdoc = service_doc(services_coll, sid)
            _p("  service %s -> %s", sid, sdoc)
            
            service_type = get_service_type(sdoc)
            if not service_type:
                _p("  service %s is not a recognized safety service; skipping", sid)
                continue

            _p("  identified as service_type=%s", service_type)
            
            mpath = model_path_from_service(models_coll, sdoc)
            _p("  resolved modelPath=%s", mpath)
            if not mpath:
                _p("  WARN: service %s has no model/modelPath; skipping", sid)
                continue

            name = safety_watcher_name(camera_oid, service_type)
            desired[name] = {
                "camera_oid": camera_oid,
                "model_path": mpath,
                "service_type": service_type
            }
            _p("  desired watcher %s -> type=%s model=%s", name, service_type, mpath)

    _p("Desired safety watchers: %d", len(desired))
    return desired

def reconcile_once(client: CircusClient, db):
    desired = compute_desired(db)
    running = list_running_watchers(client)

    desired_names = set(desired.keys())
    
    # Get all running safety watchers (any that match our prefixes)
    all_prefixes = [config["prefix"] for config in SERVICE_CONFIGS.values()]
    running_safety = {w for w in running if any(w.startswith(prefix) for prefix in all_prefixes)}

    to_add = sorted(desired_names - running_safety)
    to_remove = sorted(running_safety - desired_names)

    _p("Reconcile: add=%s remove=%s", to_add, to_remove)

    for name in to_add:
        spec = desired[name]
        add_safety_watcher(client, spec["camera_oid"], spec["model_path"], spec["service_type"])

    for name in to_remove:
        stop_and_rm_watcher(client, name)

def main():
    _p("camera_config_reconciler starting endpoint=%s mongo=%s/%s", CIRCUS_ENDPOINT, MONGO_URI, MONGO_DB)
    _p("Supported safety services: %s", list(SERVICE_CONFIGS.keys()))
    
    # marker so you can see this process actually started
    try:
        open(os.path.join(LOG_DIR, "camera_config_reconciler.started"), "a").close()
    except Exception:
        pass

    client = CircusClient(endpoint=CIRCUS_ENDPOINT, timeout=10)
    mongo = MongoClient(MONGO_URI)
    db = mongo[MONGO_DB]

    # Initial reconcile
    reconcile_once(client, db)

    # Change streams + periodic safety net
    pipeline = [{'$match': {'operationType': {'$in': ['insert', 'update', 'replace', 'delete']}}}]
    last_full = time.time()

    while True:
        try:
            with db["cameraConfig"].watch(pipeline, full_document="updateLookup") as stream:
                for _ in stream:
                    reconcile_once(client, db)
                    if time.time() - last_full > 300:
                        last_full = time.time()
                        reconcile_once(client, db)
        except Exception as e:
            _p("Change stream error: %s (retry in 5s)", e)
            time.sleep(5)
            try:
                reconcile_once(client, db)
            except Exception as e2:
                _p("reconcile after error failed: %s", e2)

if __name__ == "__main__":
    main()