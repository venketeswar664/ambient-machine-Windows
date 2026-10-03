import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField

class Stores(db.Document):
    store_id = db.StringField()
    name = db.StringField()
    organization = db.StringField()
    format = db.StringField()
    port = db.IntField()
    category = db.StringField()
    state = db.StringField()
    city = db.StringField()
    district = db.StringField()
    location = db.DictField()
    layout = db.DictField()
    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updatedAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))


class Zones(db.Document):
    zone_id = db.StringField(required=True, unique=True) 
    name = db.StringField(required=True)  
    zone_type = db.StringField(required=True)  
    store = db.ReferenceField('Stores')  
    colourHex = db.StringField(default="#09467c")  
    roi = db.ListField(db.ListField(db.FloatField())) 
    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updatedAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))

    
class Services(db.Document):
    service_name = db.StringField(required=True, unique=True)
    descriptions= db.StringField()
    pipeline_path = db.StringField(required=True)
    fixed_zones = db.BooleanField(default=False)
    zones = db.ListField(db.ReferenceField('Zones'))  
    created_at = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    
class DefaultServices(db.Document):
    name = db.StringField(required=True, unique=True)
    descriptions= db.StringField()
    pipeline_path = db.StringField(required=True)
    fixed_zones = db.BooleanField(default=False)
    zones = db.ListField(db.ReferenceField('Zones'))   
    created_at = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))


class Cameras(db.Document):
    device_name = db.StringField(required=True, unique=True)  # Format: Camera-001
    store = db.ReferenceField("Stores")  # Store associated with this camera
    camera_address = db.StringField(required=True)  # Video feed URL or location
    services = db.ReferenceField("Services")
    zones = db.ListField(db.ReferenceField('Zones'))  # References multiple zone mappings
    process_skip_frame = db.IntField(default=5)  # Skip frame interval for processing
    active = db.BooleanField(default=True)
    save_plotted_video = db.BooleanField(default=False)
    save_raw_video = db.BooleanField(default=False)
    rotation = db.IntField(default=0)  # Camera rotation in degrees
    status = db.StringField()
    department = db.StringField()
    frame = db.StringField()

    created_at = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updated_at = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    meta = {
        'collection': 'cameras'
    }
    def __str__(self):
        return f"Camera {self.device_name}"
      
    


class Metadata(Document):
    # Basic Video and Frame Info
    frame_number = IntField(required=True)
    evidence_frame_number = StringField()
    time_stamp = DateTimeField(required=True)
    video_path = StringField(required=False)
    evidence_path = StringField(required=True)
    device_name = StringField(required=True)
    device = StringField()
    inference_time = FloatField()
    raw_frame_path = StringField()
    plotted_frame_path = StringField()
    # Track ID Information (nested dictionary with string keys)
    track_ids_info = DictField(
        field=DictField(
            fields={
                "track_id": StringField(required=True),
                "track_id_path_list": StringField(required=True),
                "bbox": ListField(IntField(), required=True),
                "confidence": FloatField(required=True),
                "label": IntField(required=True),
                "label_name": StringField(required=True),
                "instance_dict": DictField()
            }
        )
    )
    meta = {
        'collection': 'metadata',
        'indexes': ['time_stamp'],
        'strict': False
    }
    
class Security(Document):
    camera = ReferenceField('Cameras')
    time_stamp = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    lights = BooleanField(required=True)
    camera_tampering = BooleanField(required=True)
    smoke = BooleanField(required=True)
    fire = BooleanField(required=True)



# === Schemas defined locally (specific to this script's logic) ===
class TrackIDMetadata(Document):
    meta = {'collection': 'track_id_metadata'} # Ensure this collection name is correct
    track_id = StringField(required=True)
    device = StringField(required=True)
    attended = IntField() # Assuming these fields exist
    dwell_time = LongField(required=True) # Assuming these fields exist
    start_time = DateTimeField(required=True)
    end_time = DateTimeField(required=True)
    gender = StringField()
    label = StringField()
    zone = StringField(required=True)
    interacted = IntField(required=True) # Assuming these fields exist
    interacted_list = ListField(StringField(), default=[]) # Assuming these fields exist
    # Add any other fields from TrackIDMetadata that are actually used by the script implicitly

# MODIFIED TrackIdCropsRun Schema
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
                "track_metadata_start_time_utc": DateTimeField(),
                "track_metadata_end_time_utc": DateTimeField(),
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



class TrackIDRecord(Document):
    track_id = StringField(required=True) 
    camera = ReferenceField(Cameras, required=True) 
    videos = ListField(StringField()) # Changed: Now a list of strings
    track_id_path_lists = ListField(ListField(StringField())) # Changed: Now a list of strings

    model_name = StringField(required=True) 
    label = StringField(required=True)
    start_time = DateTimeField() # Will store UTC datetime
    end_time = DateTimeField()   # Will store UTC datetime
    zones = DictField(field=DictField(field=ObjectIdField()))
    created_at = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)) # UTC default
    updated_at = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)) # UTC default

    meta = {
        'collection': 'track_id_records',
        'indexes': [
            {'fields': ('track_id', 'camera', 'model_name'), 'unique': True},
            'camera',
            'model_name',
            'videos', # Index on list of strings
            'start_time',
            'end_time',
        ],
        'auto_create_index': True,
        'ordering': ['-updated_at']
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.datetime.now(datetime.timezone.utc) # Ensure updated_at is UTC
        return super(TrackIDRecord, self).save(*args, **kwargs)

    def __str__(self):
        cam_name = self.camera.name if self.camera and hasattr(self.camera, 'name') else 'N/A'
        return f"TrackIDRecord(track_id='{self.track_id}', camera='{cam_name}', model='{self.model_name}')"

    

    
    

