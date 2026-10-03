# src/core/state_manager.py

class StateManager:
    def __init__(self, pipeline_config, data_dict, logger=None):
        self.pipeline_config = pipeline_config
        self.data_dict = data_dict
        self.logger = logger
        self.initialized_machines = {}

    def is_machine_initialized(self, machine_name):
        return machine_name in self.initialized_machines

    def add_initialized_machine(self, machine_name, machine_instance):
        self.initialized_machines[machine_name] = machine_instance

    def get_machine(self, machine_name):
        return self.initialized_machines.get(machine_name)