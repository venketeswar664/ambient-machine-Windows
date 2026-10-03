from pymongo import MongoClient
import os
import json
import logging
import time
from datetime import datetime
from bson import json_util
import requests
from datetime import timezone


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mongo-to-loki")

# MongoDB Connection
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://root:root@localhost:27017/?authSource=sentinel')
MONGO_DB = os.environ.get('MONGO_DB', 'sentinel')
MONGO_COLLECTION = os.environ.get('MONGO_COLLECTION', 'sys_logs')

# Loki Connection
LOKI_URL = os.environ.get('LOKI_URL', 'http://localhost:3100/loki/api/v1/push')

# Function to send logs to Loki
def send_log_to_loki(log_data):
    payload = {"streams": [{"stream": log_data["labels"], "values": [[log_data["timestamp"], log_data["message"]]]}]}
    
    try:
        response = requests.post(
            LOKI_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code >= 400:
            logger.error(f"Error sending logs to Loki: {response.status_code} - {response.text}")
        else:
            logger.info(f"Log sent to Loki: {log_data}")
    except Exception as e:
        logger.error(f"Error sending logs to Loki: {e}")

def process_log(log):
    try:
        log_dict = json.loads(json_util.dumps(log))

        # Extract timestamp from log (fallback to current time if missing)
        timestamp = log_dict.get('timestamp') or log_dict.get('ts') or log_dict.get('time', datetime.utcnow())
        if isinstance(timestamp, dict) and "$date" in timestamp:
            timestamp = timestamp["$date"]

        # Convert to datetime object and ensure it's in UTC
        dt_obj = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
        timestamp_ns = str(int(dt_obj.timestamp() * 1e9))  # Convert to nanoseconds

        # Extract labels (including 'category')
        labels = {
            "job": "mongodb",
            "collection": MONGO_COLLECTION
        }

        # Add category as a Loki label if it exists
        if "category" in log_dict:
            labels["category"] = str(log_dict["category"])  # Convert category to string

        # Include other metadata fields as labels
        for key in ["level", "source"]:
            if key in log_dict:
                labels[key] = str(log_dict[key])

        message = log_dict.get("message", json.dumps(log_dict))

        return {"timestamp": timestamp_ns, "labels": labels, "message": message}
    except Exception as e:
        logger.error(f"Error processing log: {e}")
        return None


# Function to watch MongoDB Change Stream
def watch_mongo_changes():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    logger.info(f"Watching MongoDB Change Stream for {MONGO_DB}.{MONGO_COLLECTION}...")

    try:
        pipeline = [{"$match": {"operationType": "insert"}}]  # Watch only new log entries
        with collection.watch(pipeline) as change_stream:
            for change in change_stream:
                log = change["fullDocument"]
                log_data = process_log(log)
                if log_data:
                    send_log_to_loki(log_data)
    except Exception as e:
        logger.error(f"Change Stream error: {e}")
        time.sleep(5)  # Wait before retrying
        watch_mongo_changes()  # Restart Change Stream

if __name__ == "__main__":
    watch_mongo_changes()
