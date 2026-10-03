import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField
from .stores_schema import Stores
from src.database.schemas.models_schema import Models

class Cameras(db.Document):
    deviceName = db.StringField(required=True)  # Format: Camera-001
    store = db.ReferenceField("Stores")  # Store associated with this camera
    cameraAddress = db.StringField(required=True)  # Video feed URL or location
    # zones = db.ListField(db.ReferenceField('Zones'))  # References multiple zone mappings
    baseModel = db.ReferenceField("Models")
    processSkipFrame = db.IntField(default=5)  # Skip frame interval for processing
    active = db.BooleanField(default=True)
    savePlottedFrame = db.BooleanField(default=False)
    saveRawFrame = db.BooleanField(default=False)
    rotation = db.IntField(default=0)  # Camera rotation in degrees
    status = db.StringField()
    department = db.StringField()
    frame = db.StringField()
    ipAddress = db.StringField()
    forwardedAddress = db.StringField()
    runId = db.IntField()

    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updatedAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    meta = {
        'strict': False,
        'collection': 'cameras'
    }
    def __str__(self):
        return f"Camera {self.device_name}"
      
    