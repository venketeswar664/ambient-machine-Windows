# src/orchestrator/orchestrator_engine.py

import json
import os
import importlib.util
from multiprocessing import Process, Manager
from threading import Thread
from src.core.machine import Machine  # Adjust the import path as necessary

class OrchestratorEngine(Machine):
    """
    The OrchestratorEngine manages the execution of a pipeline and can act as a machine.
    """

    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.logger = logger  # Store the logger
        self.pipeline_config_path = config.get('pipeline_config_path')
        self.verbose = logger.verbose if logger else False
        self.pipeline = None
        self.machines = {}
        self.execution_list = []
        self.data_dict = {}
        self.load_pipeline()
        self.initialize_machines()

    def get_data(self, input_data):
        self.data_dict.update(input_data)
        return input_data

    def process(self, data):
        if self.validate_pipeline():
            self.start_orchestration()
        else:
            if self.logger:
                self.logger.error("Pipeline validation failed.")
            raise RuntimeError("Pipeline validation failed.")

    def send_data(self, processed_data):
        output_keys = self.config.get('output', {}).keys()
        output_data = {key: self.data_dict.get(key) for key in output_keys}
        return output_data

    def store_data(self):
        pass

    def on_start(self):
        super().on_start()
        if self.logger:
            self.logger.info("OrchestratorEngine started.")

    def on_finish(self):
        super().on_finish()
        if self.logger:
            self.logger.info("OrchestratorEngine finished.")

    def load_pipeline(self):
        try:
            with open(self.pipeline_config_path, 'r') as file:
                self.pipeline = json.load(file)
            if self.logger:
                self.logger.info(f"Pipeline loaded: {self.pipeline['pipeline_name']}")
            # Replace placeholders with actual values
            self.pipeline = self.replace_placeholders(self.pipeline)
        except FileNotFoundError as e:
            if self.logger:
                self.logger.error(f"Pipeline configuration file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(f"Invalid JSON in pipeline configuration file: {e}")
            raise

    def replace_placeholders(self, config_dict):
        import re

        placeholder_pattern = re.compile(r'\{\{\s*(\w+)\s*\}\}')

        def replace(obj):
            if isinstance(obj, dict):
                return {k: replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace(v) for v in obj]
            elif isinstance(obj, str):
                matches = placeholder_pattern.findall(obj)
                for match in matches:
                    value = self.data_dict.get(match, '')
                    obj = obj.replace(f'{{{{ {match} }}}}', str(value))
                return obj
            else:
                return obj

        return replace(config_dict)

    def validate_pipeline(self):
        if self.logger:
            self.logger.info("Validating pipeline structure...")
        modules = self.pipeline['modules']
        for module_name, module_info in modules.items():
            input_fields = module_info.get('input', {})
            if input_fields:
                missing_inputs = []
                for input_key in input_fields.keys():
                    input_available = False
                    # Check if input_key is in data_dict
                    if input_key in self.data_dict:
                        input_available = True
                    else:
                        # Check previous modules' outputs
                        for prev_module_name, prev_module_info in modules.items():
                            if module_name in prev_module_info.get('next_module', []):
                                output_fields = prev_module_info.get('output', {})
                                if input_key in output_fields.keys():
                                    input_available = True
                                    break
                    if not input_available:
                        missing_inputs.append(input_key)
                if missing_inputs:
                    if self.logger:
                        self.logger.error(f"Input fields {missing_inputs} for module '{module_name}' are not provided by any previous module or data_dict.")
                    return False
        if self.logger:
            self.logger.info("Pipeline structure validated successfully!")
        return True

    def initialize_machines(self):
        if self.logger:
            self.logger.info("Initializing machines...")
        modules = self.pipeline['modules']
        for module_name, module_info in modules.items():
            machine_instance = self.load_machine(module_info)
            self.machines[module_name] = {
                'instance': machine_instance,
                'info': module_info
            }
            input_fields = module_info.get('input', {})
            if not input_fields:
                # No inputs; can execute immediately
                self.execution_list.append(module_name)
            else:
                # Check if all input fields are available in data_dict
                inputs_available = all(input_key in self.data_dict for input_key in input_fields.keys())
                if inputs_available:
                    self.execution_list.append(module_name)
        if self.logger:
            self.logger.info("Machines initialized.")


    def load_machine(self, module_info):
        module_path = module_info['module']
        config = module_info.get('config', {})
        # Use the 'name' field as the class name
        class_name = module_info['name']
        
        # Adjust the module path to be importable
        module_dir = os.path.dirname(module_path)
        if module_dir not in os.sys.path:
            os.sys.path.append(module_dir)
        
        try:
            # Dynamically import the module
            module_filename = os.path.basename(module_path)
            module_name, _ = os.path.splitext(module_filename)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            MachineClass = getattr(module, class_name)
        
            # Instantiate the machine, passing the logger to it
            machine_instance = MachineClass(config, logger=self.logger)
            return machine_instance
        except (ImportError, AttributeError) as e:
            if self.logger:
                self.logger.error(f"Failed to load machine '{class_name}' from '{module_path}': {e}")
            raise

    def start_orchestration(self):
        if self.logger:
            self.logger.info("\nStarting orchestration process...")
        while self.execution_list:
            current_execution_list = self.execution_list.copy()
            if len(current_execution_list) == 1:
                self._execute(current_execution_list)   
            elif len(current_execution_list) > 1:
                # Determine execution mode based on machine configurations
                execution_mode = self.get_execution_mode(current_execution_list)
                if execution_mode == 'multiprocessing':
                    self._execute_in_multiprocessing(current_execution_list)
                elif execution_mode == 'multithreading':
                    self._execute_in_multithreading(current_execution_list)
                else:
                    self._execute(current_execution_list)
            else:
                if self.logger:
                    self.logger.info("No machines to execute.")
            # Update execution list with next modules
            self.update_execution_list(current_execution_list)

    def get_execution_mode(self, machine_list):
        modes = set()
        for machine_name in machine_list:
            execute_config = self.machines[machine_name]['info'].get('execute', {}).get('process', {})
            if execute_config.get('multiprocessing'):
                modes.add('multiprocessing')
            elif execute_config.get('multithreading'):
                modes.add('multithreading')
            else:
                modes.add('singlethread')
        if len(modes) == 1:
            return modes.pop()
        else:
            return 'singlethread'  # Default to single-threaded if mixed modes

    def _execute(self, machine_list):
        for machine_name in machine_list:
            output_data = self.run_machine(machine_name, self.data_dict)
            self.data_dict.update(output_data)

    def _execute_in_multiprocessing(self, machine_list):
        if self.logger:
            self.logger.info("Executing machines in multiprocessing mode...")
        manager = Manager()
        shared_dict = manager.dict(self.data_dict)
        processes = []
        for machine_name in machine_list:
            p = Process(target=self.run_machine_mp, args=(machine_name, shared_dict))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        self.data_dict.update(shared_dict)

    def _execute_in_multithreading(self, machine_list):
        if self.logger:
            self.logger.info("Executing machines in multithreading mode...")
        shared_dict = self.data_dict.copy()
        threads = []
        for machine_name in machine_list:
            t = Thread(target=self.run_machine_thread, args=(machine_name, shared_dict))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        self.data_dict.update(shared_dict)

    def run_machine(self, machine_name, data_dict):
        machine_info = self.machines[machine_name]
        instance = machine_info['instance']
        info = machine_info['info']
        if self.logger:
            self.logger.info(f"Executing machine: {machine_name}")

        # Prepare input data with only the keys specified in the 'input' field
        input_data = {}
        input_keys = info.get('input', {})
        for key in input_keys.keys():
            if key in data_dict:
                input_data[key] = data_dict[key]
            else:
                input_data[key] = None  # Or handle missing keys as needed

        # Run machine lifecycle
        instance.on_start()
        instance.get_data(input_data)
        instance.process(input_data)
        output_data = instance.send_data(None)
        instance.store_data()
        instance.on_finish()

        if output_data is None:
            output_data = {}
        return output_data

    def run_machine_mp(self, machine_name, shared_dict):
        output_data = self.run_machine(machine_name, shared_dict)
        shared_dict.update(output_data)

    def run_machine_thread(self, machine_name, shared_dict):
        output_data = self.run_machine(machine_name, shared_dict)
        shared_dict.update(output_data)

    def update_execution_list(self, machine_list):
        next_modules = []
        for machine_name in machine_list:
            next_modules.extend(self.machines[machine_name]['info'].get('next_module', []))
        self.execution_list = next_modules

    def monitor_pipeline(self):
        pass
