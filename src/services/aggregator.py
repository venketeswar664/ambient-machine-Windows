import time
from datetime import datetime,  timezone
from pymongo import MongoClient
from typing import Optional
from bson import ObjectId

class TrackMetadataAggregator:
    def __init__(self, mongo_uri: str, database_name: str):
        """
        Initialize the Track Metadata Aggregator
        
        Args:
            mongo_uri: MongoDB connection string
            database_name: Name of the database
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.metadata_collection = self.db['metadata']
        self.track_metadata_collection = self.db['track_id_metadata']
        self._ensure_indexes()



    def _ensure_indexes(self):
        """Create required indexes for the track_id_metadata collection"""
        try:
            # Create unique compound index on trackId and zone
            self.track_metadata_collection.create_index(
                [("trackId", 1), ("zone", 1)],
                unique=True,
                name="trackId_zone_unique"
            )
            print("Index created/verified: trackId_zone_unique")
        except Exception as e:
            print(f"Index already exists or error creating index: {str(e)}")


    def run_aggregation(self, start_time: datetime, end_time: datetime, device: str):
        """
        Run the aggregation pipeline for track metadata
        
        Args:
            start_time: Start timestamp for filtering
            end_time: End timestamp for filtering
            device: Device identifier for filtering
        """
        try:
            pipeline = [
                # Match documents based on time range and device
                {
                    "$match": {
                        "time_stamp": {
                            "$gte": start_time,
                            "$lte": end_time
                        },
                        "device": device
                    }
                },
                # Convert track_ids_info object to array
                {
                    "$addFields": {
                        "track_ids_info": {
                            "$objectToArray": "$track_ids_info"
                        }
                    }
                },
                # Unwind the track_ids_info array
                {
                    "$unwind": {
                        "path": "$track_ids_info"
                    }
                },
                # Extract location from specific zone (belapur_entry_outside)
                {
                    "$addFields": {
                        "location": {
                            "$getField": {
                                "field": "location",
                                "input": {
                                    "$getField": {
                                        "field": "belapur_entry_outside",
                                        "input": {
                                            "$getField": {
                                                "field": "instance_dict",
                                                "input": "$track_ids_info.v"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                # Extract zone from instance_dict
                {
                    "$set": {
                        "zone": {
                            "$let": {
                                "vars": {
                                    "insideEntries": {
                                        "$filter": {
                                            "input": {
                                                "$objectToArray": {
                                                    "$ifNull": [
                                                        "$track_ids_info.v.instance_dict",
                                                        {}
                                                    ]
                                                }
                                            },
                                            "as": "entry",
                                            "cond": {
                                                "$eq": [
                                                    "$$entry.v.location",   
                                                    "outside"
                                                ]
                                            }
                                        }
                                    }
                                },
                                "in": {
                                    "$cond": {
                                        "if": {
                                            "$gt": [
                                                {
                                                    "$size": {
                                                        "$ifNull": [
                                                            "$$insideEntries",
                                                            []
                                                        ]
                                                    }
                                                },
                                                0
                                            ]
                                        },
                                        "then": {
                                            "$let": {
                                                "vars": {
                                                    "firstInside": {
                                                        "$arrayElemAt": [
                                                            "$$insideEntries",
                                                            0
                                                        ]
                                                    }
                                                },
                                                "in": {
                                                    "$arrayElemAt": [
                                                        {
                                                            "$split": [
                                                                "$$firstInside.k",
                                                                "-"
                                                            ]
                                                        },
                                                        0
                                                    ]
                                                }
                                            }
                                        },
                                        "else": None
                                    }
                                }
                            }
                        }
                    }
                },
                # Group by track_id and zone
                {
                    "$group": {
                        "_id": {
                            "track_id": "$track_ids_info.k",
                            "zone": "$zone"
                        },
                        "startTime": {
                            "$min": "$time_stamp"
                        },
                        "endTime": {
                            "$max": "$time_stamp"
                        },
                        "startFrameNumber": {
                            "$min": "$frame_number"
                        },
                        "endFrameNumber": {
                            "$max": "$frame_number"
                        },
                        "trackId": {
                            "$first": "$track_ids_info.k"
                        },
                        "device": {
                            "$first": "$device"
                        },
                        "label": {
                            "$first": "$track_ids_info.v.label"
                        },
                        "labelName": {
                            "$first": "$track_ids_info.v.label_name"
                        },
                        "zone": {
                            "$first": "$zone"
                        },
                        "location": {
                            "$first": "$location"
                        }
                    }
                },
                # Remove _id field
                {
                    "$project": {
                        "_id": 0
                    }
                },
                # Merge results into track_id_metadata collection
                {
                    "$merge": {
                        "into": "track_id_metadata",
                        "on": ["trackId", "zone"],
                        "whenMatched": "replace",
                        "whenNotMatched": "insert"
                    }
                }
            ]
            
            # Execute the aggregation pipeline
            self.metadata_collection.aggregate(pipeline)
            
            print(f"[{datetime.now()}] Aggregation completed successfully")
            print(f"Device: {device}, Start: {start_time}, End: {end_time}")
            
            # Get count of processed records
            count = self.track_metadata_collection.count_documents({
                "device": device,
                "startTime": {"$gte": start_time},
                "endTime": {"$lte": end_time}
            })
            print(f"Track metadata records: {count}")
            
            return True
            
        except Exception as e:
            print(f"[{datetime.now()}] Error during aggregation: {str(e)}")
            raise
    
    # def run_continuous(self, device: str, interval_seconds: int = 10, 
    #                   time_window_seconds: int = 60):
    #     """
    #     Run aggregation continuously at specified intervals
        
    #     Args:
    #         device: Device identifier
    #         interval_seconds: Time interval between runs (default: 10 seconds)
    #         time_window_seconds: Time window for each aggregation (default: 60 seconds)
    #     """
    #     print(f"Starting continuous aggregation for device: {device}")
    #     print(f"Interval: {interval_seconds}s, Time window: {time_window_seconds}s")
    #     print("Press Ctrl+C to stop\n")
        
    #     try:
    #         while True:
    #             end_time = datetime.now()
    #             start_time = datetime.fromtimestamp(
    #                 end_time.timestamp() - time_window_seconds
    #             )
                
    #             self.run_aggregation(start_time, end_time, device)
                
    #             time.sleep(interval_seconds)
                
    #     except KeyboardInterrupt:
    #         print("\n\nStopping aggregation...")
    #     finally:
    #         self.close()
    
    def run_once(self, start_time: datetime, end_time: datetime, device: str):
        """
        Run aggregation once for specified parameters
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            device: Device identifier
        """
        try:
            result = self.run_aggregation(start_time, end_time, device)
            return result
        finally:
            self.close()
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print("MongoDB connection closed")


# Example usage
if __name__ == "__main__":
    # Configuration
    MONGO_URI = "mongodb://sentinel:sentinelMongo_test123@10.8.0.26:27017/sentinel_warehouse?directConnection=true&authSource=admin"  # Update with your MongoDB URI
    DATABASE_NAME = "sentinel_warehouse"      # Update with your database name
    DEVICE = ObjectId("6933dc543b1fc78128b96abc")                     # Update with your device identifier
    
    # Initialize aggregator
    aggregator = TrackMetadataAggregator(MONGO_URI, DATABASE_NAME)
    
    # Option 1: Run continuously every 10 seconds
    # aggregator.run_continuous(
    #     device=DEVICE,
    #     interval_seconds=10,
    #     time_window_seconds=60  # Look back 60 seconds
    # )
    
    # Option 2: Run once for specific time range
    # from datetime import timedelta
    # end_time = datetime.now()
    # start_time = end_time - timedelta(seconds=60)
    start_time = datetime(2025, 12, 6, 9, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2025, 12, 6, 21, 0, 0, tzinfo=timezone.utc)

# Run aggregation once
    aggregator.run_once(start_time, end_time, DEVICE)
