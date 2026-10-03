# src/core/machine.py

import sys
import traceback
import tracemalloc
import time
import psutil

try:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False
from abc import ABC, abstractmethod
from collections import deque

class Machine(ABC):

    def __init__(self, machine_name, state_manager):
        self.machine_name = machine_name
        self.state_manager = state_manager
        self.pipeline_config = state_manager.pipeline_config
        self.config = self.pipeline_config['pipeline'][self.machine_name]
        self.data_dict = state_manager.data_dict
        self.logger = state_manager.logger
        self.next_machines = []
    

        self.verbose = 1  # 0: No print, 1: logs print, 2: logs + data print

        # Register this machine as initialized
        self.state_manager.add_initialized_machine(self.machine_name, self)

        # Get execution methods and their configurations
        self.methods_config = self.config.get('execute', {})
        

        # Get output configuration
        self.output_fields = self.config.get('output', {})

        self.initialize_next_machines()

    def initialize_next_machines(self):
        next_machine_names = self.config.get('next_module', [])
        for next_machine_name in next_machine_names:
            # Check if the next machine has already been initialized
            if self.state_manager.is_machine_initialized(next_machine_name):
                if self.logger:
                    self.logger.debug(f"Machine '{next_machine_name}' already initialized. Adding to next_machines.")
                # Get the initialized machine from the state manager
                next_machine = self.state_manager.get_machine(next_machine_name)
                self.next_machines.append(next_machine)
                continue  # Skip re-initialization

            next_machine_info = self.pipeline_config['pipeline'].get(next_machine_name)
            if not next_machine_info:
                if self.logger:
                    self.logger.error(f"Next machine '{next_machine_name}' not found in pipeline configuration.")
                continue

            module_path = next_machine_info['module']
            class_name = next_machine_info['name']

            # Dynamically import the next machine class
            try:
                module = __import__(module_path.replace('/', '.').replace('.py', ''), fromlist=[class_name])
                next_machine_class = getattr(module, class_name)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error importing {class_name} from {module_path}: {e}")
                continue

            # Initialize the next machine with the same state_manager
            next_machine = next_machine_class(
                machine_name=next_machine_name,
                state_manager=self.state_manager
            )
            self.next_machines.append(next_machine)

    def execute(self):
        # Use a queue for iterative execution
        queue = deque()
        queue.append(self)

        while queue:
            current_machine = queue.popleft()

            current_machine.on_start()
            try:
                # Replace placeholders in config
                current_machine.replace_placeholders()

                # Execute each method in the 'execute' config
                for method_name, method_config in current_machine.methods_config.items():
                    method = getattr(current_machine, method_name, None)
                    if not method:
                        error_message = f"Method '{method_name}' not found in {current_machine.machine_name}"
                        if current_machine.logger:
                            current_machine.logger.error(error_message)
                        raise AttributeError(error_message)
                    # Get method-specific input data
                    # print(method_config)
                    # print( method_name)
                    method_input_data = current_machine.get_method_input(method_config.get('input', {}))
                    # print(f"Method input data for {method_name}: {method_input_data}")
                    # Execution starts
                    # if self.logger:
                    #     self.logger.info(f"Execution of '{method_name}' started.")
                    # print(f"Executing method: {method_name} with input data: {method_input_data}")

                    # start_time = time.time()
                    # tracemalloc.start()
                    # processed_data = method(method_input_data)
                    # current, peak = tracemalloc.get_traced_memory()
                    # tracemalloc.stop()
                    # elapsed_time = time.time() - start_time
                    # # Execution finished
                    # if self.logger:
                    #     self.logger.info(f"Execution of '{method_name}' finished in {elapsed_time:.4f} seconds.")
                    #     self.logger.info(f"Memory usage: Current = {current / 10**6:.4f} MB; Peak = {peak / 10**6:.4f} MB.")




                    # Initialize NVML for GPU memory monitoring
                    gpu_mem_info_avail = False
                    if _HAS_NVML:
                        try:
                            nvmlInit()
                            gpu_mem_info_avail = True
                        except Exception:
                            pass

                    try:
                        start_time = time.time()
                        tracemalloc.start()
                        
                        # Execute the method
                        processed_data = method(method_input_data)
                        # print(f"Processed data: {processed_data}")
                        # Memory tracking
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        elapsed_time = time.time() - start_time

                        # RAM usage
                        process = psutil.Process()
                        ram_current = process.memory_info().rss / (1024 ** 2)  # Resident memory in MB
                        ram_peak = process.memory_info().vms / (1024 ** 2)  # Virtual memory in MB (Peak-like)

                        # GPU usage
                        gpu_current = 0.0
                        gpu_peak = 0.0
                        if gpu_mem_info_avail:
                            try:
                                gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assumes GPU 0; adjust for multi-GPU setups
                                gpu_mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
                                gpu_current = gpu_mem_info.used / (1024 ** 2)  # Convert bytes to MB
                                gpu_peak = gpu_mem_info.total / (1024 ** 2)    # Total available memory as peak (approximation)
                            except Exception:
                                pass

                        # Log execution details
                        if self.logger:
                            self.logger.info(f"Execution of '{method_name}' finished in {elapsed_time:.4f} seconds.")
                            self.logger.info(f"Memory usage: Current = {current / 10**6:.4f} MB; Peak = {peak / 10**6:.4f} MB.")
                            self.logger.info(f"RAM usage: Current = {ram_current:.4f} MB; Peak = {ram_peak:.4f} MB.")
                            self.logger.info(f"GPU usage: Current = {gpu_current:.4f} MB; Peak = {gpu_peak:.4f} MB.")
                    finally:
                        # Shutdown NVML
                        if gpu_mem_info_avail:
                            try:
                                nvmlShutdown()
                            except Exception:
                                pass

                    # Send processed data
                    current_machine.send_data(processed_data)
                    # print(f"Data sent from {current_machine.machine_name}")
                # Enqueue next machines
                    # print(f"Next machines to execute: {[next_machine.machine_name for next_machine in current_machine.next_machines]}")
                for next_machine in current_machine.next_machines:
                    self.logger.info(f"Next machine '{next_machine.machine_name}' added to queue.")
                    queue.append(next_machine)

            except Exception as e:
                if current_machine.logger:
                    current_machine.logger.error(f"Exception in {current_machine.machine_name}: {e}")
                    current_machine.logger.error(traceback.format_exc())
                else:
                    print(f"Exception in {current_machine.machine_name}: {e}")
                    traceback.print_exc()
            finally:
                current_machine.on_finish()

    def get_method_input(self, input_fields):
        print(f"\n***************************************  {self.machine_name}  *********************************")

        input_data = {}
        missing_keys = []
        print("input_fields: ", input_fields)
        for key in input_fields.keys():
            if key in self.data_dict:
                input_data[key] = self.data_dict[key]
                if self.verbose == 2:
                    print(f"{self.machine_name} Got data:")
                    self.logger.info(f"{key}: {input_data[key]}")
            else:
                missing_keys.append(key)
        if missing_keys:
            error_message = f"Missing required input data for keys: {missing_keys} in method of machine {self.machine_name}"
            if self.logger:
                self.logger.error(error_message)
            raise ValueError(error_message)
        # Log input data keys
        if self.logger:
            self.logger.info(f"Input data keys: {list(input_data.keys())}")
        return input_data

    @abstractmethod
    def process(self, input_data):
        pass  # Must be implemented in subclasses

    def send_data(self, processed_data):
        sent_keys = []
        for key in self.output_fields.keys():
            if key in processed_data:
                self.data_dict[key] = processed_data[key]
                sent_keys.append(key)
                if self.verbose == 2:
                    print("Sending data:")
                    self.logger.info(f"{key}: {processed_data[key]}")
        # Log sent data keys
        if self.logger and sent_keys:
            self.logger.info(f"Sent data keys: {sent_keys}")

    def replace_placeholders(self):
        # Replace placeholders in config with actual data from data_dict
        def replace(obj):
            if isinstance(obj, dict):
                return {k: replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace(item) for item in obj]
            elif isinstance(obj, str):
                # Check for placeholders
                import re
                placeholders = re.findall(r'\{\{\s*(.*?)\s*\}\}', obj)
                for placeholder in placeholders:
                    if placeholder in self.data_dict:
                        obj = obj.replace(f'{{{{ {placeholder} }}}}', str(self.data_dict[placeholder]))
                return obj
            else:
                return obj

        self.config = replace(self.config)

    def on_start(self):
        if self.logger:
            self.logger.info(f"{self.machine_name} started.")

    def on_finish(self):
        if self.logger:
            self.logger.info(f"{self.machine_name} finished.\n")
