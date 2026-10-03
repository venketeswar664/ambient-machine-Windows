import json
"""
Add config file
"""
def load_config(config_path):
    # Open the JSON file in read mode
    with open(config_path, "r") as json_file:
        # Load the JSON data into a Python dictionary
        config = json.load(json_file)
    
    # # Now, config_data contains the contents of the JSON file as a dictionary
    # print(config)
    return config


BASE_CONFIG_DATA = None

def read_base_config():
    global BASE_CONFIG_DATA  # Declare BASE_CONFIG_DATA as a global variable

    if BASE_CONFIG_DATA==None:
        # Specify the file path
        config_path = "src/configs/config.json"

        # Open the JSON file in read mode
        with open(config_path, "r") as json_file:
            # Load the JSON data into a Python dictionary
            BASE_CONFIG_DATA = json.load(json_file)
        
        # Now, BASE_config_data contains the contents of the JSON file as a dictionary
        # print(BASE_CONFIG_DATA)
        
    return BASE_CONFIG_DATA
