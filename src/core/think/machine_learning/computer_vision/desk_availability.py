# employee_counts.py

import logging
import pandas as pd
import numpy as np
from mongoengine import Document, DateTimeField, FloatField

logging.basicConfig(level=logging.INFO)


class EmployeeCounts(Document):
    """
    MongoEngine schema for storing employee count KPI results.
    """
    time_window = DateTimeField(required=True)
    ppe_kit_employees = FloatField(required=True)
    normal_employees = FloatField(required=True)
    total_employees = FloatField(required=True)

    meta = {
        'collection': 'employee_counts',
        'indexes': ['time_window']
    }


def calculate_employee_counts(metadata_df, zone_name='Scrape', time_window_minutes=1):
    """
    Calculate the maximum number of PPE kit employees (label 13) and normal employees (label 4)
    detected in the specified zone for each time window.

    Saves results into MongoDB 'employee_counts' collection and returns a DataFrame.
    """
    logging.info("Calculating Employee Counts KPI...")
    try:
        # Ensure timestamps are in datetime format
        metadata_df['time_stamp'] = pd.to_datetime(metadata_df['time_stamp'])

        # Create time windows
        metadata_df['time_window'] = metadata_df['time_stamp'].dt.floor(f'{time_window_minutes}min')

        metrics = []

        # Group data by time windows
        for time_window, window_data in metadata_df.groupby('time_window'):
            # Initialize counts for each frame in the time window
            frame_counts = []

            for zones_data in window_data['zones_track_id']:
                if zone_name in zones_data:
                    # Count PPE kit employees (label 13) and normal employees (label 4)
                    ppe_employees = len({
                        entry['track_id'] for entry in zones_data[zone_name]
                        if entry['label'] == 13.0
                    })
                    normal_employees = len({
                        entry['track_id'] for entry in zones_data[zone_name]
                        if entry['label'] == 4.0
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

        # Save metrics to MongoDB
        for metric in metrics:
            EmployeeCounts(**metric).save()

        return pd.DataFrame(metrics)

    except Exception as e:
        logging.error(f"Error calculating employee counts: {e}")
        return pd.DataFrame()
