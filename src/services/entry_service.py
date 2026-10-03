import sys
import time
import schedule
# Import timedelta and time for dynamic date calculation
from datetime import datetime, timedelta, time as datetime_time
from pymongo import MongoClient
from bson.objectid import ObjectId
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Constants ---
DB_NAME = "sentinel_warehouse"
COLLECTION_NAME = 'metadata'
# This cron string is for "every minute" - you may want to change this
# to run once a day, e.g., "30 0 * * *" for 12:30 AM
CRON_SCHEDULE = "* * * * *"

# --- Main Aggregation Pipeline ---
# This function is unchanged
def get_pipeline(start_day, end_day, device_id):
    """Returns the main aggregation pipeline for a given time range and device."""
    return [
        {
            "$match": {
                "time_stamp": {
                    "$gte": start_day,
                    "$lt": end_day
                },
                "device": device_id
            }
        },
        {
            "$addFields": {
                "track_ids_info": {
                    "$objectToArray": "$track_ids_info"
                }
            }
        },
        {
            "$unwind": {
                "path": "$track_ids_info",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$set": {
                "track_ids_info.v.instance_dict": {
                    "$arrayToObject": {
                        "$map": {
                            "input": {
                                "$objectToArray": "$track_ids_info.v.instance_dict"
                            },
                            "as": "entry",
                            "in": {
                                "k": {
                                    "$switch": {
                                        "branches": [
                                            {
                                                "case": {
                                                    "$regexMatch": {
                                                        "input": "$$entry.k",
                                                        "regex": "inside",
                                                        "options": "i"
                                                    }
                                                },
                                                "then": "inside"
                                            },
                                            {
                                                "case": {
                                                    "$regexMatch": {
                                                        "input": "$$entry.k",
                                                        "regex": "outside",
                                                        "options": "i"
                                                    }
                                                },
                                                "then": "outside"
                                            }
                                        ],
                                        "default": "$$entry.k"
                                    }
                                },
                                "v": "$$entry.v"
                            }
                        }
                    }
                }
            }
        },
        {
            "$addFields": {
                "required": {
                    "$cond": [
                        {
                            "$or": [
                                {
                                    "$and": [
                                        {
                                            "$regexMatch": {
                                                "input": "$track_ids_info.v.instance_dict.outside.location",
                                                "regex": "outside",
                                                "options": "i"
                                            }
                                        },
                                        {
                                            "$regexMatch": {
                                                "input": "$track_ids_info.v.instance_dict.inside.location",
                                                "regex": "outside",
                                                "options": "i"
                                            }
                                        }
                                    ]
                                },
                                {
                                    "$eq": [
                                        "$track_ids_info.v.instance_dict",
                                        None
                                    ]
                                }
                            ]
                        },
                        False,
                        True
                    ]
                }
            }
        },
        {
            "$match": {
                # "track_ids_info.v.label_name": {
                #   "$regex": "truck",
                #   "$options": "i"
                # },
                "required": True
            }
        },
        {
            "$addFields": {
                "zone_condition": "$track_ids_info.v.instance_dict.inside.location"
            }
        },
        {
            "$sort": {
                "time_stamp": 1
            }
        },
        {
            "$group": {
                "_id": "$track_ids_info.k",
                "startTime": {
                    "$min": "$time_stamp"
                },
                "endTime": {
                    "$max": "$time_stamp"
                },
                "zones": {
                    "$push": "$zone_condition"
                },
                "device": {
                    "$first": "$device"
                },
                "label": {
                    "$first": "$track_ids_info.v.label_name"
                }
            }
        },
        {
            "$sort": {
                "_id": 1
            }
        },
        {
            "$addFields": {
                "zones": {
                    "$reduce": {
                        "input": "$zones",
                        "initialValue": [],
                        "in": {
                            "$concatArrays": [
                                "$$value",
                                {
                                    "$cond": {
                                        "if": {
                                            "$eq": [
                                                {
                                                    "$arrayElemAt": [
                                                        "$$value",
                                                        -1
                                                    ]
                                                },
                                                "$$this"
                                            ]
                                        },
                                        "then": [],
                                        "else": ["$$this"]
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        },
        {
            "$addFields": {
                "entry": {
                    "$and": [
                        {
                            "$eq": [
                                {
                                    "$first": "$zones"
                                },
                                "outside"
                            ]
                        },
                        {
                            "$eq": [
                                {
                                    "$last": "$zones"
                                },
                                "inside"
                            ]
                        }
                    ]
                },
                "exit": {
                    "$and": [
                        {
                            "$eq": [
                                {
                                    "$first": "$zones"
                                },
                                "inside"
                            ]
                        },
                        {
                            "$eq": [
                                {
                                    "$last": "$zones"
                                },
                                "outside"
                            ]
                        }
                    ]
                },
                "timeStanded": {
                    "$dateDiff": {
                        "startDate": "$startTime",
                        "endDate": "$endTime",
                        "unit": "second"
                    }
                }
            }
        },
        {
            "$match": {
                "$expr": {
                    "$and": [
                        {
                            "$gt": [
                                {
                                    "$size": "$zones"
                                },
                                1
                            ]
                        },
                        {
                            "$gt": ["$timeStanded", 1]
                        }
                    ]
                }
            }
        },
        {
            "$match": {
                "entry": True
            }
        },
        {
            "$project": {
                "_id": 0,
                "startTime": 1,
                "endTime": 1,
                "device": 1,
                "trackID": "$_id",
                "service": ObjectId(
                    "68e8dcbc2d39ca526c037994"
                ),
                "keyword": "warehouseEntry",
                "extraInfo": {
                    "label": "$label"
                }
            }
        },
        {
            "$merge": {
                "into": "nexus_test",
                "on": [
                    "trackID",
                    "startTime",
                    "device",
                    "service"
                ],
                "whenMatched": "merge",
                "whenNotMatched": "insert"
            }
        }
    ]

# --- Camera Fetching Pipeline ---
# This function is unchanged
def get_cameras_pipeline():
    """Returns the pipeline to fetch active cameras with the specific service."""
    return [
        {
            "$match": {
                "active": {"$ne": False}
            }
        },
        {
            "$project": {
                "_id": 1
            }
        },
        {
            "$lookup": {
                "from": "cameraConfig",
                "localField": "_id",
                "foreignField": "cameraOid",
                "as": "config",
                "pipeline": [
                    {
                        "$sort": {
                            "_id": -1
                        }
                    },
                    {
                        "$limit": 1
                    },
                    {
                        "$unwind": "$services"
                    },
                    {
                        "$match": {
                            "services.service": ObjectId("68e8dcbc2d39ca526c037994"),
                        }
                    }
                ]
            }
        },
        {
            "$match": {
                "config": {"$ne": []}
            }
        },
        {
            "$project": {
                "_id": 1,
            }
        }
    ]

# --- Helper for Threading ---
# This function is unchanged
def process_camera(camera, db, start_date, end_date):
    """
    Runs the aggregation pipeline for a single camera.
    This function is designed to be run in a separate thread.
    """
    camera_id = camera['_id']
    try:
        pipeline_to_run = get_pipeline(start_date, end_date, camera_id)
        # We use list() to force the aggregation (lazy cursor) to execute
        list(db[COLLECTION_NAME].aggregate(pipeline_to_run))
        return f"✅ Successfully processed camera: {camera_id}"
    except Exception as e:
        return f"❌ FAILED to process camera {camera_id}: {e}"

# --- Main Aggregation Function (MODIFIED) ---
def run_aggregation(client):
    """
    Fetches all active cameras and runs the aggregation pipeline
    for each one in parallel.
    """
    try:
        db = client[DB_NAME]
        
        # 1. Get the list of cameras
        camera_pipeline = get_cameras_pipeline()
        cameras_list = list(db["cameras"].aggregate(camera_pipeline))

        if not cameras_list:
            print(f"[{datetime.now().isoformat()}] ℹ️ No active cameras found for service. Skipping.")
            return

        # 2. Set time range to YESTERDAY 2:30 PM to 11:59:59 PM
        # 2. Set time range for ALL OF TODAY
        now = datetime.now()
        today_date = now.date()
        
        # Start time is 00:00:00 today (midnight)
        start_date = datetime.combine(today_date, datetime_time(0, 0, 0))
        
        # Get tomorrow's date
        tomorrow_date = today_date + timedelta(days=1)
        
        # End time is 00:00:00 tomorrow (which works as $lt)
        end_date = datetime.combine(tomorrow_date, datetime_time(0, 0, 0))

        print(f"[{datetime.now().isoformat()}] 🚀 Starting aggregation for {len(cameras_list)} cameras for range: {start_date} to {end_date}")

        # 3. Run aggregations in parallel (equivalent to Promise.allSettled)
        # ... (rest of your function) ...

        # print(f"[{datetime.now().isoformat()}] 🚀 Starting aggregation for {len(cameras_list)} cameras for range: {start_date} to {end_date}")

        # 3. Run aggregations in parallel (equivalent to Promise.allSettled)
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Create a future for each camera processing task
            futures = {
                executor.submit(process_camera, camera, db, start_date, end_date): camera
                for camera in cameras_list
            }
            
            # As tasks complete, print their results
            for future in as_completed(futures):
                result = future.result()
                print(f"[{datetime.now().isoformat()}] {result}")

        print(
            f"[{datetime.now().isoformat()}] ✅ All camera aggregations finished for time range staring {start_date.isoformat()}."
        )
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] 🚨 Aggregation failed: {e}")
    # Note: The client is NOT closed here, as it's managed by the main script
    # for the cron job to reuse.

# --- Main Execution (MODIFIED) ---
if __name__ == "__main__":
    # Hardcoded URI as requested
    URI = "mon  godb://sentinel:sentinelMongo_test123@10.8.0.26:27017/sentinel_warehouse"

    # Create a single, persistent client for the service
    try:
        client = MongoClient(URI, maxPoolSize=50, minPoolSize=5)
        # Test connection
        client.server_info()
        print(f"[{datetime.now().isoformat()}] MongoDB connection successful.")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Failed to connect to MongoDB: {e}")
        sys.exit(1)

    # Schedule the job (equivalent to cron.schedule)
    # Note: The JS cron string "* * * * *" is "every minute"
    schedule.every().minute.do(run_aggregation, client=client)

    print(f"[{datetime.now().isoformat()}] 🏃‍♂️ Running initial aggregation on start...")
    run_aggregation(client)  # Run once immediately, like in the JS script

    print(f"[{datetime.now().isoformat()}] ⏰ Scheduler started. Will run every minute.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().isoformat()}] Shutting down service...")
    finally:
        client.close()
        print(f"[{datetime.now().isoformat()}] MongoDB connection closed. Exiting.")