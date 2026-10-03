import mongoengine as db 
from mongoengine import Document, StringField, FloatField, ListField, IntField, BooleanField, DictField , ReferenceField
from datetime import datetime
# from .services_schema import Services

class Models(db.Document):
    # device_name = db.StringField(required=True, unique=True) 
    modelType = db.StringField(required=True)
    modelName = db.StringField(requested=True)  
    modelPath = db.StringField(required=True)  
    trackerPath = db.StringField()
    convertEngine = db.BooleanField(default=True)
    classIds =  db.ListField(db.IntField())
    conf = db.FloatField(default=0.6)  
    inputDims =  db.ListField(db.IntField())
    store = db.ReferenceField("Stores")
    # services = db.ListField(db.ReferenceField("Services"))
    createdAt = StringField(default=datetime.utcnow)
    updatedAt = StringField(default=datetime.utcnow)
    configYamlPath = StringField(requested=False, default="")


    meta = {
        'collection': 'models',
        'indexes': [
            'modelName',
            'modelType',
            'createdAt'
        ]
    }
