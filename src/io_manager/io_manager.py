import importlib.util
import inspect
import os

from src.io_manager import DeviceStateEnum, DeviceState, Union
from src.utils.config import read_base_config, load_config

BASE_CONFIG = read_base_config()
IO_MANAGER_CONFIG = load_config(BASE_CONFIG["io_manager"])


class IOManager:
    """
    Class responsible for managing devices and interactions in the IO system.
    """

    subclasses = list()
    initialized_devices = dict()

    def __init__(self) -> None:
        # Initialize orchestrator instance
        # self.init_all_io_devices()
        pass

    def init_all_io_devices(self):

        """
        Initializes all sensors and actuators iteratively.
        Returns:
            list: A list containing instances of all initialized sensors and actuators.
        """
        sensors_ = IO_MANAGER_CONFIG.get("sensors").values()
        # print(f"IO_MANAGER_CONFIG: {sensors_}")
        # These args kwargs will be used and updated later as per the requirements form IO device 
        sensors = [sensor for sensor in [dev_type for dev_type in IO_MANAGER_CONFIG.get("sensors").values()]]
        sensors_to_init = [sensor_device for sensor_device in sensors if sensor_device['launch']]

        actuators = [actuator for actuator in [dev_type for dev_type in IO_MANAGER_CONFIG.get("actuators").values()]]
        actuators_to_init = [actuator_device for actuator_device in actuators if actuator_device['launch']]

        self.devices_to_init = list()
        self.devices_to_init.extend(sensors_to_init)
        self.devices_to_init.extend(actuators_to_init)
        # print(f"Intitializing: {devices_to_init}")

        for device in self.devices_to_init:
            try:
                # print("Initialzing device:", device)
                module_path = device["module_path"]
                module_name = os.path.basename(module_path).rstrip(".py")

                # Loading the module spec
                spec = importlib.util.spec_from_file_location(module_name, module_path)

                # Loading the module
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                classes = inspect.getmembers(module, inspect.isclass)
                class_names = [class_info[0] for class_info in classes if class_info[1].__module__ == module_name]
                class_object = getattr(module, class_names[0])

                device_config = load_config(device["conf"])
                # Now ye can use the loaded instance
                device_instance = class_object(device_config)
                device["instance"] = device_instance

                # example:
                # device_instance.connect()

                self.initialized_devices[device["name"]] = device
            
            except Exception as e:
                print(f"ERROR: while initializing io device: {device}. \n {e}")
        
        # print(f"INITIALIZED:  {self.initialized_devices}")
            
        return self.initialized_devices
    
    def get_initialized_devices(self):
        return self.initialized_devices
    
    # def __init__(self) -> None:
    #     self.state: DeviceState = DeviceState(state_value=DeviceStateEnum.DISCONNECTED)

    def set_state(self, new_state_value: Union[DeviceStateEnum, str]) -> None:
        """
        Set the state of the device without validation.
        
        Args:
            new_state_value (Union[DeviceStateEnum, str]): New state value.
        """
        self.state = DeviceState(state_value=new_state_value)

    def update_state(self, new_state_value: DeviceStateEnum) -> None:
        """
        Update the device state with validation.
        
        Args:
            new_state_value (DeviceStateEnum): New state value.
        """
        self.state = DeviceState(state_value=new_state_value)
