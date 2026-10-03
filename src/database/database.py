from mongoengine import connect, disconnect

# from src.database.schemas.inference_result import InferenceResult
from src.utils.config import read_base_config, load_config
import emoji

BASE_CONFIG = read_base_config()
DATABASE_CONFIG = load_config(BASE_CONFIG["database"])

class Database:
    
    def __init__(self) -> None:
        db_config = DATABASE_CONFIG
        #get the credentials information fro t he config file

        self.db_url = db_config["db_url"]
        self.project = db_config["project"]
        self.mode = db_config["mode"]
        

    def connect_db(self):
        try:
            disconnect()
            connect(host=self.db_url, connectTimeoutMS=1000,  serverSelectionTimeoutMS=1000)
            # print("url", db_url)
            print(f"Database connected {emoji.emojize(':thumbs_up:')} for project {self.project} with {self.mode} mode")
            return True
        
        except Exception as e:
            print(f"Unable to connect database {emoji.emojize(':thumbs_down:')} for project {self.project} with {self.mode} mode")
            return False