# employee_counts.py

import logging
from typing import Optional

import pandas as pd
import numpy as np
from mongoengine import Document, DateTimeField, FloatField, connect, NotUniqueError, MultipleObjectsReturned

logging.basicConfig(level=logging.INFO)

# Establish connection to MongoDB (Adjust parameters as needed)
connect(db='your_database_name', host='localhost', port=27017)  # Replace with your MongoDB connection details


class EmployeeCounts(Document):
    """
    MongoEngine schema for storing employee count KPI results.
    """
    time_window = DateTimeField(required=True, unique=True)  # Ensure uniqueness to prevent duplicates
    ppe_kit_employees = FloatField(required=True)
    normal_employees = FloatField(required=True)
    total_employees = FloatField(required=True)

    meta = {
        'collection': 'employee_counts',
        'indexes': ['time_window']
    }


def calculate_employee_counts(metadata_df: pd.DataFrame, zone_name: str = 'Scrape', time_window_minutes: int = 1) -> pd.DataFrame:
    """
    Calculate the maximum number of PPE kit employees (label 13) and normal employees (label 4)
    detected in the specified zone for each time window.

    Args:
        metadata_df: DataFrame containing metadata records
        zone_name: Name of the zone to analyze for employee counts
        time_window_minutes: Size of time window in minutes

    Saves results into MongoDB 'employee_counts' collection and returns a DataFrame.

    Parameters:
        metadata_df (pd.DataFrame): DataFrame containing metadata records.
        zone_name (str): The name of the zone to analyze. Default is 'Scrape'.
        time_window_minutes (int): The size of the time window in minutes. Default is 1.

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics.
    """
    logging.info(f"Calculating Employee Counts KPI for zone '{zone_name}'...")
    try:
        # Ensure necessary columns are present
        required_columns = {'time_stamp', 'zones_track_id'}
        if not required_columns.issubset(metadata_df.columns):
            missing = required_columns - set(metadata_df.columns)
            raise ValueError(f"Missing required columns in metadata_df: {missing}")

        # Ensure timestamps are in datetime format
        metadata_df['time_stamp'] = pd.to_datetime(metadata_df['time_stamp'])

        # Create time windows by flooring the timestamps
        metadata_df['time_window'] = metadata_df['time_stamp'].dt.floor(f'{time_window_minutes}T')

        metrics = []

        # Group data by time windows
        for time_window, window_data in metadata_df.groupby('time_window'):
            # Initialize counts for each frame in the time window
            frame_counts = []

            for zones_data in window_data['zones_track_id']:
                if isinstance(zones_data, dict) and zone_name in zones_data:
                    # Ensure zones_data[zone_name] is iterable
                    zone_entries = zones_data.get(zone_name, [])
                    if isinstance(zone_entries, list):
                        # Count unique PPE kit employees (label 13) and normal employees (label 4)
                        ppe_employees = len({
                            entry['track_id'] for entry in zone_entries
                            if entry.get('label') == 13.0
                        })
                        normal_employees = len({
                            entry['track_id'] for entry in zone_entries
                            if entry.get('label') == 4.0
                        })
                        frame_counts.append((ppe_employees, normal_employees))

            # Calculate maximum counts for the time window
            if frame_counts:
                max_ppe = max(count[0] for count in frame_counts)
                max_normal = max(count[1] for count in frame_counts)
            else:
                max_ppe = 0
                max_normal = 0

            metrics.append({
                'time_window': time_window,
                'ppe_kit_employees': max_ppe,
                'normal_employees': max_normal,
                'total_employees': max_ppe + max_normal
            })

        # Convert metrics to DataFrame for potential further processing
        metrics_df = pd.DataFrame(metrics)

        # Prepare bulk insertion
        employee_count_docs = [
            EmployeeCounts(
                time_window=row['time_window'],
                ppe_kit_employees=row['ppe_kit_employees'],
                normal_employees=row['normal_employees'],
                total_employees=row['total_employees']
            )
            for _, row in metrics_df.iterrows()
        ]

        # Perform bulk insertion with error handling to avoid duplicates
        for doc in employee_count_docs:
            try:
                doc.save()
                logging.info(f"Saved EmployeeCounts for window {doc.time_window}")
            except NotUniqueError:
                logging.warning(f"EmployeeCounts for window {doc.time_window} already exists. Skipping.")
            except Exception as e:
                logging.error(f"Error saving EmployeeCounts for window {doc.time_window}: {e}")

        return metrics_df

    except Exception as e:
        logging.error(f"Error calculating employee counts: {e}", exc_info=True)
        return pd.DataFrame()
