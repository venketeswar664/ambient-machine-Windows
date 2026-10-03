# src/core/sensors/mongo_machine.py

import os
from bson import ObjectId

from src.core.machine import Machine
from src.database.schemas.cameras_schema import Cameras
from src.database.schemas.zones_schema import Zones
from src.database.schemas.stores_schema import Stores
from src.database.database import Database
from src.monitoring_stack.mongodb_logger import initialize_logger


class MongoMachine(Machine):
    """
    MongoMachine retrieves active camera documents from the database.

    It queries the specified collection for active documents based on the provided query
    and passes the processed data to the next machine.
    """

    def __init__(self, machine_name, state_manager):
        super().__init__(machine_name, state_manager)
        """
        Initializes the MongoMachine.

        Args:
            machine_name (str): The name of the machine in the pipeline.
            state_manager: State manager for managing pipeline states.
        """
        config = self.config.get("config", {})
        self.collection_path = config.get("path")  # Module path for the collection
        self.collection_name = config.get("collection")  # Name of the collection class
        self.query = config.get("query", {})  # Query to execute

        # Extract filter and projection from query
        self.query_filter = self.query.get("filter", {})
        self.projection = self.query.get("projection", {})

        # Initialize database connection
        self.database = Database()
        self.database.connect_db()
        print(f"Connected to database: {self.database.db_url}")
         # Initialize logger with machine-specific category
        self.logger = initialize_logger(category="mainpipeline")


    def get_data(self):
        """
        Retrieves input data.

        Since this machine does not require any input data, it simply returns an empty dictionary.

        Returns:
            dict: Empty dictionary.
        """
        return super().get_data()  # Default implementation returns {}

    def process(self, input_data):
        """
        Processes the data by querying active documents from the collection.
        """
        try:
            module_path = self.collection_path.replace("/", ".").replace(".py", "")
            module = __import__(module_path, fromlist=[self.collection_name])
            Collection = getattr(module, self.collection_name)

            # Start with the filter from config (e.g. any extra conditions from main_pipeline.json)
            effective_filter = dict(self.query_filter)

            # Always enforce active=True — only process cameras that are currently active.
            # This is hardcoded here so it cannot be accidentally removed from config.
            effective_filter["active"] = True

            # HARDCODED store filter for testing — only process cameras belonging to this store.
            # TODO: Replace with os.environ.get("STORE_ID") when deploying to multiple machines.
            store_id = "6a3d7f02cc1c4c00c19dd6f4"
            effective_filter["store"] = ObjectId(store_id)

            print(f"Querying collection '{self.collection_name}' with filter: {effective_filter}")

            active_documents = Collection.objects(__raw__=effective_filter).only(*self.projection.keys())

            docs_list = list(active_documents)
            print(f"Found {len(docs_list)} document(s) matching filter (active=True, store={store_id}).")

            documents_list = [doc.to_mongo().to_dict() for doc in docs_list][:]

            if self.logger:
                self.logger.info(f"Retrieved {len(documents_list)} active camera(s) for store='{store_id}'.")

            return {"active_cameras": documents_list}


        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving documents: {e}")
            raise


    def send_data(self, processed_data):
        """
        Sends the processed data to the shared data_dict.

        Args:
            processed_data (dict): Processed data containing active documents.
        """
        super().send_data(processed_data)
