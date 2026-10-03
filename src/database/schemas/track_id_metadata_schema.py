import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField


class TrackIDMetadata(Document):
    meta = {'collection': 'track_id_metadata'} 
    track_id = StringField(required=True)
    device = StringField(required=True)
    attended = IntField() 
    dwell_time = LongField(required=True)
    start_time = DateTimeField(required=True, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    end_time = DateTimeField(required=True, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    gender = StringField()
    label = StringField()
    zone = StringField(required=True)
    interacted = IntField(required=True) 
    interacted_list = ListField(StringField(), default=[])
