import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField
from .cameras_schema import Cameras

class Zones(db.Document):
    name = db.StringField(required=True)  
    zoneType = db.StringField(required=True)  
    cameraOid = db.ReferenceField('Cameras')  
    colourHex = db.StringField(default="#09467c")  
    roi = db.ListField(db.ListField(db.FloatField())) 
    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updatedAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))

    # meta = {
    #     'indexes': [
    #         { 'unique': True}
    #     ]
    # }