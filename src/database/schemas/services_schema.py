import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField
from src.database.schemas.models_schema import Models

class Services(db.Document):
    name = db.StringField(required=True)
    descriptions= db.StringField()
    pipelinePath = db.StringField(required=True)
    fixedZones = db.ListField(db.StringField(), required=False)
    model = db.ReferenceField("Models")
    default = db.ListField(db.BooleanField()) 
    keywords = db.ListField(db.StringField())
    createdAt = db.DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))