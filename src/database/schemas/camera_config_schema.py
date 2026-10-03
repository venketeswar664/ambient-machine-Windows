import mongoengine as db
import datetime
from src.database.schemas.zones_schema import Zones
from src.database.schemas.models_schema import Models
from src.database.schemas.services_schema import Services
from src.database.schemas.cameras_schema import Cameras
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField


class ServiceItem(db.EmbeddedDocument):
    service = db.ReferenceField("Services", required=True)
    zones = db.ListField(db.ReferenceField("Zones"), required=True)

class CameraConfig(db.Document):
    services = db.EmbeddedDocumentListField(ServiceItem, required=True)
    timeStamp = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    cameraOid = db.ReferenceField("Cameras")

    meta = {
        'collection': 'cameraConfig',
        'indexes': ['timeStamp', 'cameraOid']
    }
