



import importlib
import os
import inspect
from src.utils.config import load_config
from src.utils.detection_manager import DetectionWindowManager
import time


class MachineLearning:

    def __init__(self, camera, zones, pipeline_json):
        self.camera = camera
        self.zones = zones

        self.input_dict = {}

        self.config = {}
        self.config["zones"] = self.zones
        self.config["camera"] = self.camera
        self.fps = camera.fps
        self.device_name = self.camera.device_name

        self.pipeline_json = pipeline_json
        self.pipeline = {}

        self.initialize_modules()

    def initialize_modules(self):

        # load pipeline
        self.stage_json_config = load_config(self.pipeline_json[0])
        # initialization prints 

        stage_name = list(self.stage_json_config.keys())[0]

        # print("\n")
        # print(f"Initializing {stage_name} pipeline.")

        # pipeline
        self.stage_config = self.stage_json_config[stage_name]
        self.pipeline_config = self.stage_config["pipeline"]
        print("Pipeline used", stage_name)
        print("camera fps", self.camera.fps)
        # if stage_name == "Face Recognition":
        #     detection_manager = DetectionWindowManager()
        print("Initializing Modules", list(self.pipeline_config.keys()))

        for module_key, module_config in self.pipeline_config.items():

            # load module
            module_object = self.initialize_class(module_config)
            self.pipeline[module_key] = module_object


    def initialize_class(self, module_config):

        # print("module_config", module_config)
        module_path = module_config["module"]
        module_name = os.path.basename(module_path).rstrip(".py")

        # Load the Module Specification
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        # Load the Module
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get Class Names from the Module
        classes = inspect.getmembers(module, inspect.isclass)
        class_names = [class_info[0] for class_info in classes if class_info[1].__module__ == module_name]
        class_object = getattr(module, class_names[0])

        device_config = module_config["config"]

        if type(device_config) == str:
            device_config = load_config(device_config)

        device_config.update(self.config)

        instance = class_object(device_config)

        return instance
    
    def update_input(self):
        frame = self.camera.rgb_img
        frame_number =  self.camera.frame_number 
        input_dict = {
            "frame": frame,
            "frame_number": frame_number,
            "camera_fps": self.camera.fps
        }

        return input_dict



    def process(self, shared_dict):

        print("\nProcessing the modules ...")
        # get frames to process 
        self.input_dict.update(self.update_input())

        # start frame processing in pipeline
        for count, (module_key, module_config) in enumerate(self.pipeline_config.items()):
            
            start_time = time.time()

            # get all the process list in the module
            process_list = module_config["process"]

            # get initialized module objects 
            instance = self.pipeline[module_key]

            # process all processes in the process_list
            for process_name in process_list:
                print("processing", process_name)
                # get method object without calling
                method = getattr(instance, process_name)

                # get input data keys from config
                input_data_keys = module_config["input"]

                # select input dict key
                _input_dict = {key: self.input_dict[key] for key in input_data_keys if key in self.input_dict}

                # Check if the method exists and is callable
                if callable(method):
                    # run the module with input data dict
                    output_dict = method(_input_dict)

                    # update input dict
                    if output_dict is not None:
                        self.input_dict.update(output_dict)


            if not set(list(output_dict.keys())) == set(module_config["output"]):

                print("Expected data keys:",  module_config["output"])

                print("Got output data keys: ", list(output_dict.keys()))

                break

            process_time = time.time() - start_time
            print(f"Processed module: ({count}) {module_key} in {process_time} s.")
            # print("data keys: ", self.input_dict.keys())
            print("\n")

        # once processing completes
        # store data and plotted(if required) frames for the process 
            
        # print("input_dict", self.input_dict)
        shared_dict[self.device_name]["data"] = self.input_dict


