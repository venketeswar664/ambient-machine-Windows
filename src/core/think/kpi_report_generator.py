import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
import pandas as pd
import os


class KPIReportGenerator:
    """
    Class to generate a report by processing KPI documents from three collections:
      - PeopleKpiCounts (people_kpi_counts)
      - UnsafeMHEKpiCounts (unsafe_mhe_kpi_counts)
      - TruckMovementCounts (truck_movement_counts)

    It calculates all tracking parameters for a given date (default: today) and
    exports them to an Excel file.
    """

    def __init__(self, db_url, db_name="sentinel"):
        """
        Initialize the KPIReportGenerator with MongoDB connection.

        :param db_url: MongoDB connection URL
        :param db_name: Name of the database to use
        """
        self.db_client = MongoClient(db_url)
        self.db = self.db_client[db_name]
        self.logger = logging.getLogger(self.__class__.__name__)

        # Define the tracking parameters (excluding "No" and "Areas")
        # Each parameter can have multiple cameras, and we'll produce separate columns for each camera.
        self.tracking_parameters = [
            {
                "Tracking Parameter": "PPE Kit Adherence (Helmet)",
                "Cameras": ["D04", "D03"],
                "UoM": "No of Incidents",
                "Remarks": "Without Helmets",
                "CalcFunc": self.calc_ppe_kit_helmet,
            },
            {
                "Tracking Parameter": "PPE Kit Adherence (Safety Vest)",
                "Cameras": ["D04", "D03"],
                "UoM": "No of Incidents",
                "Remarks": "Without Safety Vest",
                "CalcFunc": self.calc_ppe_kit_safety_vest,
            },
            {
                "Tracking Parameter": "Unsafe MHE movements",
                "Cameras": ["D04"],
                "UoM": "No of Incidents",
                "Remarks": "Accidents/near miss for machine-to-machine",
                "CalcFunc": self.calc_unsafe_mhe,
            },

            {
                "Tracking Parameter": "Truck Idle time",
                "Cameras": ["D03"],
                "UoM": "Mins",
                "Remarks": "Truck idle time on the dock",
                "CalcFunc": self.calc_truck_idle_time,
            },
            {
                "Tracking Parameter": "Dock Utilization",
                "Cameras": ["D03"],
                "UoM": "Mins",
                "Remarks": "Total dock occupied hours",
                "CalcFunc": self.calc_dock_utilization,
            },
            {
                "Tracking Parameter": "No of vehicles unloaded",
                "Cameras": ["D03"],
                "UoM": "Numbers",
                "Remarks": "Number of Trucks on the dock",
                "CalcFunc": self.calc_vehicles_unloaded,
            },
            {
                "Tracking Parameter": "Pallets unloaded per truck",
                "Cameras": ["D03"],
                "UoM": "Number of pallets per truck",
                "Remarks": "",
                "CalcFunc": self.calc_pallets_unloaded_per_truck,
            },
            {
                "Tracking Parameter": "Unloading TAT per truck",
                "Cameras": ["D03"],
                "UoM": "Mins",
                "Remarks": "Turn around time of truck per truck",
                "CalcFunc": self.calc_unloading_tat_per_truck,
            },
            {
                "Tracking Parameter": "Idle Manpower",
                "Cameras": ["D04", "D03"],
                "UoM": "Man-hours",
                "Remarks": "Total idle manpower time",
                "CalcFunc": self.calc_idle_manpower,
            },
        ]

    def process_kpi_documents(self, date=None):
        """
        Process KPI documents for the specified date (or today).
        :param date: Date for which to process KPI documents. Defaults to today.
        """
        if date is None:
            date = datetime.today().date()

        self.logger.info(f"Processing KPI documents for date: {date}")

        # We'll construct a row for each KPI (i.e., each item in tracking_parameters),
        # but for each camera we create a separate column in that row.
        results = []

        for param in self.tracking_parameters:
            row = {
                "Tracking Parameter": param["Tracking Parameter"],
                "UoM": param["UoM"],
                "Remarks": param["Remarks"],
            }
            # For each camera in param["Cameras"], run the aggregator
            for cam in param["Cameras"]:
                # We'll suffix the column name with the camera, e.g. "Number_D04", "Number_D03"
                col_name = f"{date}_{cam}"
                row[col_name] = param["CalcFunc"](date, [cam])

            results.append(row)

        file_path = self.export_to_excel(results, date)

        return file_path

    # -------------------------------------------------------------------------
    #  PEOPLE KPI CALCULATIONS
    # -------------------------------------------------------------------------
    def calc_ppe_kit_helmet(self, date, cameras):
        """Summation of no_helmet_track_ids from people_kpi_counts for the given cameras."""
        return self._aggregate_people_kpi(date, cameras, field="no_helmet_track_ids")

    def calc_ppe_kit_safety_vest(self, date, cameras):
        """Summation of no_ppe_kit_track_ids from people_kpi_counts for the given cameras."""
        return self._aggregate_people_kpi(date, cameras, field="no_ppe_kit_track_ids")

    def calc_idle_manpower(self, date, cameras):
        """Summation of idle_track_ids from people_kpi_counts for the given cameras."""
        return self._aggregate_people_kpi(date, cameras, field="idle_track_ids")

    def _aggregate_people_kpi(self, date, cameras, field):
        """
        Utility aggregator for people_kpi_counts.
        Counts all unique keys in a dict field (e.g., no_helmet_track_ids)
        where the value is greater than 10.
        """
        coll = self.db["people_kpi_counts"]
        start_dt = datetime.combine(date, datetime.min.time())
        end_dt = datetime.combine(date, datetime.max.time())

        query = {
            "camera_name": {"$in": cameras},
            "time_window": {"$gte": start_dt, "$lt": end_dt},
            field: {"$ne": {}},  # Ensure the field is non-empty
        }

        pipe = [
            {"$match": query},  # Filter documents matching the query
            {"$project": {
                "filtered_keys": {
                    "$filter": {
                        "input": {"$objectToArray": f"${field}"},
                        "as": "item",
                        "cond": {"$gt": ["$$item.v", 10]}  # Only include keys with value > 10
                    }
                }
            }},
            {"$unwind": "$filtered_keys"},  # Flatten the array of filtered keys
            {"$group": {
                "_id": None,
                "unique_keys": {"$addToSet": "$filtered_keys.k"}  # Collect unique keys
            }},
            {"$project": {
                "total_unique_keys": {"$size": "$unique_keys"}  # Count unique keys
            }},
        ]

        result = list(coll.aggregate(pipe))
        if result:
            return result[0]["total_unique_keys"]
        return 0


    # -------------------------------------------------------------------------
    # UNSAFE MHE KPI CALCULATIONS
    # -------------------------------------------------------------------------
    def calc_unsafe_mhe(self, date, cameras):
        """
        Reads from unsafe_mhe_kpi_counts -> unsafe_movement.
        Counts unique keys in unsafe_movement where the value exceeds 10.
        """
        coll = self.db["unsafe_mhe_kpi_counts"]
        start_dt = datetime.combine(date, datetime.min.time())
        end_dt = datetime.combine(date, datetime.max.time())

        query = {
            "camera_name": {"$in": cameras},
            "time_window": {"$gte": start_dt, "$lt": end_dt},
            "unsafe_movement": {"$ne": {}},
        }

        pipe = [
            {"$match": query},
            {
                "$project": {
                    "valid_keys": {
                        "$filter": {
                            "input": {"$objectToArray": "$unsafe_movement"},
                            "as": "pair",
                            "cond": {"$gt": ["$$pair.v", 10]},  # Filter where value > 10
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "unique_keys": {"$addToSet": "$valid_keys.k"},  # Collect unique keys
                }
            },
            {
                "$project": {
                    "unique_count": {"$size": "$unique_keys"},  # Count unique keys
                }
            },
        ]

        result = list(coll.aggregate(pipe))
        if result:
            return result[0]["unique_count"]
        return 0


    # -------------------------------------------------------------------------
    # TRUCK MOVEMENT KPI CALCULATIONS
    # -------------------------------------------------------------------------

    def calc_truck_kpis(self, date, cameras):
        """
        Truck Idle Time (Minutes):
        Calculate the total minutes when no people_entering and people_exiting were recorded,
        and truck_present_count is more than 50% of total_docs.
        """
        coll = self.db["truck_movement_counts"]
        start_dt = datetime.combine(date, datetime.min.time())
        end_dt = datetime.combine(date, datetime.max.time())

        query = {
            "camera_name": {"$in": cameras},
            "time_window": {"$gte": start_dt, "$lt": end_dt},
        }

        pipe = [
                {"$match": query},
                {
                    "$sort": {
                        "_id": 1
                    }
                },
                {
                    "$addFields": {
                        "isTruckPresent": {
                            "$gt": ["$truck_present_count", 0]
                        }
                    }
                },
                {
                    "$setWindowFields": {
                        "partitionBy": None,
                        "sortBy": {
                            "time_window": 1
                        },
                        "output": {
                            "previousTruckPresent": {
                                "$shift": {
                                    "output": "$isTruckPresent",
                                    "by": -1
                                }
                            },
                            "previousTimeWindow": {
                                "$shift": {
                                    "output": "$time_window",
                                    "by": -1
                                }
                            }
                        }
                    }
                },
                {
                    "$match": {
                        "previousTruckPresent": {
                            "$ne": None
                        }
                    }
                },
                {
                    "$addFields": {
                        "transition": {
                            "$cond": {
                                "if": {
                                    "$ne": [
                                        "$isTruckPresent",
                                        "$previousTruckPresent"
                                    ]
                                },
                                "then": 1,
                                "else": 0
                            }
                        }
                    }
                },
                {
                    "$setWindowFields": {
                        "sortBy": {
                            "time_window": 1
                        },
                        "output": {
                            "cumulativeTransitionGroup": {
                                "$sum": "$transition",
                                "window": {
                                    "documents": ["unbounded", "current"]
                                }
                            }
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$cumulativeTransitionGroup",
                        "timestamps": {
                            "$push": "$time_window"
                        },
                        "isTruckPresent": {
                            "$first": "$isTruckPresent"
                        },
                        "pallets_unloaded": {
                            "$sum": "$objects_exiting"
                        },
                        "idle_minutes": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$expr": {
                                            "$and": [
                                                {
                                                    "$eq": ["$people_exiting", 0]
                                                },
                                                {
                                                    "$eq": ["$people_entering", 0]
                                                }
                                            ]
                                        }
                                    },
                                    1,
                                    0
                                ]
                            }
                        }
                    }
                },
                {
                    "$match": {
                        "isTruckPresent": True
                    }
                },
                {
                    "$sort": {
                        "_id": 1
                    }
                },
                {
                    "$addFields": {
                        "firstTime": {
                            "$arrayElemAt": ["$timestamps", 0]
                        },
                        "lastTime": {
                            "$arrayElemAt": ["$timestamps", -1]
                        }
                    }
                },
                {
                    "$addFields": {
                        "totalTime": {
                            "$dateDiff": {
                                "startDate": "$firstTime",
                                "endDate": "$lastTime",
                                "unit": "minute"
                            }
                        }
                    }
                },
                {
                    "$match": {
                        "$expr": {
                            "$and": [
                                {
                                    "$gt": ["$totalTime", 5]
                                },
                                {
                                    "$lt": ["$totalTime", 600]
                                }
                            ]
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "pallets_unloaded": 1,
                        "unloading_tat": "$totalTime",
                        "idle_minutes": 1
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "trucks_unloaded": {"$sum": 1},
                        "trucks": {"$push": "$$ROOT"}
                    }
                },
                {
                    "$project": {
                        "_id": 0
                    }
                }
            ]

        result = list(coll.aggregate(pipe))
        if result:
            print("Truck kpis:", result)
            return result

        return 0

    def calc_truck_idle_time(self, date, cameras):
        self.truck_kpis = self.calc_truck_kpis(date=date, cameras=cameras)

        if self.truck_kpis == 0:
            return 0

        print("self.truck_kpis", self.truck_kpis)
        # count idel time
        _truck_kpi = self.truck_kpis[0]["trucks"]
        total_idle_time = 0

        if _truck_kpi:
            for truck in _truck_kpi:
                total_idle_time += truck["idle_minutes"]

        return total_idle_time

    def calc_dock_utilization(self, date, cameras):
               # count idel time

        if self.truck_kpis == None:
            self.truck_kpis = self.calc_truck_kpis(date=date, cameras=cameras)


        if self.truck_kpis == 0:
            return 0
        
        _truck_kpi = self.truck_kpis[0]["trucks"]
        dock_utilization_time = 0

        if _truck_kpi:
            for truck in _truck_kpi:
                dock_utilization_time += truck["unloading_tat"]

        return dock_utilization_time


    def calc_vehicles_unloaded(self, date, cameras, metadata=False):
        if self.truck_kpis == None:
            self.truck_kpis = self.calc_truck_kpis(date=date, cameras=cameras)

        if self.truck_kpis == 0:
            return 0
        vehicles_count = self.truck_kpis[0]["trucks_unloaded"]

        return vehicles_count




    def calc_pallets_unloaded_per_truck(self, date, cameras):
        if self.truck_kpis == None:
            self.truck_kpis = self.calc_truck_kpis(date=date, cameras=cameras)

        if self.truck_kpis == 0:
            return 0
        _truck_kpi = self.truck_kpis[0]["trucks"]
        pallets_counts = 0
        count = 0
        if _truck_kpi:
            for truck in _truck_kpi:
                pallets_counts += truck["pallets_unloaded"]
                count += 1

        return pallets_counts / count




    def calc_unloading_tat_per_truck(self, date, cameras):
        if self.truck_kpis == None:
            self.truck_kpis = self.calc_truck_kpis(date=date, cameras=cameras)

        if self.truck_kpis == 0:
            return 0
        
        _truck_kpi = self.truck_kpis[0]["trucks"]
        truck_tat_per_truck = 0
        count = 0
        if _truck_kpi:
            for truck in _truck_kpi:
                truck_tat_per_truck += truck["unloading_tat"]
                count += 1

        return truck_tat_per_truck / count



    # -------------------------------------------------------------------------
    # Export to Excel
    # -------------------------------------------------------------------------
    def export_to_excel(self, results, date):
        """
        Export the results to an Excel file.
        Each row is one KPI, plus separate columns for each camera's number.
        :param results: List of dictionaries containing results.
        :param date: Date for which the report is generated.
        """

        directory = f"./results/videos/FM/{date}/Report/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = f"{directory}KPI_Report_{date}.xlsx"
        df = pd.DataFrame(results)

        with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="KPI Report")

        self.logger.info(f"Report successfully exported to {file_name}")
        return file_name


def run(date=None):
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    logging.basicConfig(level=logging.INFO)
    from src.database.database import Database

    database = Database()
    # Example usage
    db_url = database.db_url
    db_name = "sentinel"

    kpi_report_generator = KPIReportGenerator(db_url, db_name)
    # This will process the KPI data for the date 16-1-2025
    file_path = kpi_report_generator.process_kpi_documents(date)
    return file_path

# ------------------------- Example Usage --------------------------------------
if __name__ == "__main__":
    date = datetime(2025, 1, 17)

    run()
