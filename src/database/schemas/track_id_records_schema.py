import mongoengine as db
import datetime
from mongoengine import Document, IntField, DateTimeField, StringField, DictField, FloatField, ListField, ReferenceField, BooleanField, LongField, ObjectIdField


class TrackIDRecord(Document):
    track_id = StringField(required=True) 
    camera = ReferenceField(Cameras, required=True) 
    videos = ListField(StringField())
    track_id_path_lists = ListField(ListField(StringField())) 
    model_name = StringField(required=True) 
    label = StringField(required=True)
    start_time = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)) 
    end_time = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc))   
    zones = DictField(field=DictField(field=ObjectIdField()))
    created_at = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = DateTimeField(default=lambda: datetime.datetime.now(datetime.timezone.utc)) 

    meta = {
        'collection': 'track_id_records',
        'indexes': [
            {'fields': ('track_id', 'camera', 'model_name'), 'unique': True},
            'camera',
            'model_name',
            'videos', 
            'start_time',
            'end_time',
        ],
        'auto_create_index': True,
        'ordering': ['-updated_at']
    }

    def save(self, *args, **kwargs):
        self.updated_at = datetime.datetime.now(datetime.timezone.utc)
        return super(TrackIDRecord, self).save(*args, **kwargs)

    def __str__(self):
        cam_name = self.camera.name if self.camera and hasattr(self.camera, 'name') else 'N/A'
        return f"TrackIDRecord(track_id='{self.track_id}', camera='{cam_name}', model='{self.model_name}')"
