
import logging
import datetime
import mongoengine as me
from src.utils.config import read_base_config, load_config

BASE_CONFIG = read_base_config()
DATABASE_CONFIG = load_config(BASE_CONFIG["database"])
db_config = DATABASE_CONFIG
db_url = db_config["db_url"]
me.connect(host=db_url)

class LogEntry(me.Document):
    """Schema for each log entry stored as a separate document."""
    timestamp = me.DateTimeField(required=True)  # Removed default timestamp, will set it dynamically
    category = me.StringField(required=True)  # "mainpipeline", "device1", "device2"
    level = me.StringField(required=True, choices=["INFO", "WARNING", "ERROR", "DEBUG"])
    message = me.StringField(required=True)

    meta = {"collection": "sys_logs"}  # Collection name

    @classmethod
    def create_log(cls, category, level, message, timestamp):
        """Insert a new log entry into MongoDB."""
        log_entry = cls(category=category, level=level, message=message, timestamp=timestamp)
        log_entry.save()
        return log_entry

class MongoDBHandler(logging.Handler):
    """Custom MongoDB Logging Handler for structured logging"""

    def __init__(self, category="mainpipeline"):
        """
        Initialize MongoDB logging handler.

        :param category: Category for logs (e.g., "mainpipeline", "device1", "device2")
        """
        super().__init__()
        self.category = category  # Default category if not passed

    def emit(self, record):
        """Send log entry to MongoDB"""
        try:
            timestamp = datetime.datetime.now(datetime.timezone.utc)  # Dynamically set timestamp to current time (UTC)
            log_entry = {
                "timestamp": timestamp,
                "level": record.levelname,
                "message": record.getMessage(),
                "category": getattr(record, "category", self.category),
            }

            LogEntry.create_log(
                category=log_entry["category"],
                level=log_entry["level"],
                message=log_entry["message"],
                timestamp=log_entry["timestamp"],  # Pass the dynamic timestamp
            )
        except Exception:
            # Ignore MongoDB connection errors to prevent spamming logs
            pass

def initialize_logger(category="mainpipeline"):
    """
    Initialize a logger with MongoDB storage.
    
    :param category: The category under which logs are stored (e.g., mainpipeline, device1, etc.)
    """
    logger = logging.getLogger(category)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers (to avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add MongoDB Logging Handler
    mongo_handler = MongoDBHandler(category=category)
    logger.addHandler(mongo_handler)

    return logger
