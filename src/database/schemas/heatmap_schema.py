# -*- coding: utf-8 -*-
'''
Created on 03-June-2024 15:02
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''

from mongoengine import Document, StringField, DateTimeField

class Heatmap(Document):
    camera_name = StringField(required=True)
    date = DateTimeField(required=True, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    heatmap_path = StringField(required=True)  # Store the path to the heatmap image

    meta = {
        'indexes': [
            {'fields': ['camera_name', 'date'], 'unique': True}
        ]
    }
