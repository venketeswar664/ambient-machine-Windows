# -*- coding: utf-8 -*-
'''
Created on 21-September-2024 20:34
Project: ambient-machine 
@author: Pranjal Bhaskare
@email: pranjalab@neophyte.live
'''


# Centralized Data Storage Class
class DatabaseHandler:
    def __init__(self, storage_type="mongo"):
        """
        Initializes the DataStore class to handle data storage in MongoDB or S3.
        
        Args:
        - storage_type (str): The type of storage to use ("mongo" or "s3").
        """
        self.storage_type = storage_type

    def store(self, data):
        """
        Stores data based on the storage type.
        
        Args:
        - data (dict): The data to be stored.
        """
        if self.storage_type == "mongo":
            self.store_in_mongo(data)
        elif self.storage_type == "s3":
            self.store_in_s3(data)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

    def store_in_mongo(self, data):
        """Placeholder method for storing data in MongoDB."""
        print(f"Storing data in MongoDB: {data}")
        # Implement MongoDB storage logic here

    def store_in_s3(self, data):
        """Placeholder method for storing data in S3."""
        print(f"Storing data in S3: {data}")
        # Implement S3 storage logic here
