import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField

class TrackIdCropsRun(Document):
    meta = {'collection': 'track_id_crops_runs'}
    run_start_timestamp = DateTimeField(required=True) # Known at start
    
    # Fields modified to allow null initially
    run_end_timestamp = DateTimeField(null=True)
    processing_duration_seconds = FloatField(null=True)
    query_window_start_utc = DateTimeField(null=True)
    query_window_end_utc = DateTimeField(null=True)
    
    query_end_time_ist_str = StringField()
    query_duration_minutes = IntField()
    archive_blob_url = StringField(null=True)
    processed_track_ids_info = DictField(
        field=DictField(
            fields={
                "track_metadata_start_time_utc": DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)),
                "track_metadata_end_time_utc": DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)),
                "device": StringField(),
                "zone": StringField(),
                "crop_relative_paths_by_timestamp": DictField(
                    field=StringField()
                )
            }
        ),
        default={} # Provide a default empty dict
    )
    status = StringField(default="pending")
    error_message = StringField(null=True)
    total_track_ids_found = IntField(default=0)
    total_metadata_entries_processed = IntField(default=0)
    total_crops_generated = IntField(default=0)
