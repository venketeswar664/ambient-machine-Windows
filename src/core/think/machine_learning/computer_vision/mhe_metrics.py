# mhe_metrics.py

import logging
import pandas as pd
from mongoengine import Document, DateTimeField, BooleanField

logging.basicConfig(level=logging.INFO)


class MHEMetrics(Document):
    """
    MongoEngine schema for storing MHE (Material Handling Equipment) KPI results.
    """
    time_window = DateTimeField(required=True)
    mhe_available = BooleanField(required=True)
    mhe_active = BooleanField(required=True)

    meta = {
        'collection': 'mhe_metrics',
        'indexes': ['time_window']
    }


def calculate_mhe_metrics(metadata_df, zone_name='HDR:23-24', label_id=10.0, time_window_minutes=1):
    """
    Calculate MHE availability and activity metrics per time window.

    Args:
        metadata_df: DataFrame containing metadata records
        zone_name: Name of the zone to analyze for MHE activity
        label_id: Label ID used to identify MHE equipment
        time_window_minutes: Size of time window in minutes

    Saves results into MongoDB 'mhe_metrics' collection.
    """
    logging.info(f"Calculating MHE Metrics KPI for zone '{zone_name}'...")
    try:
        metadata_df['time_stamp'] = pd.to_datetime(metadata_df['time_stamp'])
        metadata_df['time_window'] = metadata_df['time_stamp'].dt.floor(f'{time_window_minutes}min')
        metrics = []

        for time_window, window_data in metadata_df.groupby('time_window'):
            mhe_available = False
            mhe_active = False

            for _, row in window_data.iterrows():
                zones_data = row.get('zones_track_id', {})
                zone_tracks = zones_data.get(zone_name, [])

                for track in zone_tracks:
                    label = track.get('label', None)
                    if label == label_id:
                        # MHE is found in this frame/time window
                        mhe_available = True
                        # Example condition for "activity"
                        if track.get('iou', 0) > 0.2:
                            mhe_active = True

            metrics.append({
                'time_window': time_window,
                'mhe_available': mhe_available,
                'mhe_active': mhe_active
            })

        # Save results to DB
        for metric in metrics:
            MHEMetrics(**metric).save()

    except Exception as e:
        logging.error(f"Error calculating MHE metrics: {e}")
