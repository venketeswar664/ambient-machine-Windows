# actuator.py

from machine import Machine

class ActuatorMachine(Machine):
    """
    ActuatorMachine performs actions based on processed data.

    This class is responsible for taking processed data from the pipeline and performing
    actions such as sending notifications, triggering alerts, or controlling hardware devices.
    """

    def __init__(self, config, logger=None):
        """
        Initializes the ActuatorMachine with configuration and logger.

        Args:
            config (dict): Configuration parameters for the actuator.
            logger (Logger, optional): Logger instance for logging events. Defaults to None.
        """
        super().__init__(config, logger)
        # Initialize actuator-specific attributes here
        self.action_type = config.get('action_type', 'print')
        # You can add more initialization as needed

    def get_data(self, input_data):
        """
        Retrieves input data for processing.

        Args:
            input_data (dict): Dictionary containing the input data required by the actuator.

        Returns:
            dict: The data to process.
        """
        if self.logger:
            self.logger.debug("ActuatorMachine received input data.")
        return input_data

    def process(self, data):
        """
        Processes the input data to determine the appropriate action.

        Args:
            data (dict): The data to process.

        Returns:
            dict: The processed data, which may include action commands.

        Raises:
            ValueError: If the input data is invalid.
        """
        if not data:
            raise ValueError("No input data provided to process.")

        # Implement logic to determine what action to take based on data
        processed_data = data  # Modify as necessary based on your logic

        if self.logger:
            self.logger.debug("Data processed by ActuatorMachine.")
        return processed_data

    def send_data(self, processed_data):
        """
        Performs the action based on processed data.

        Args:
            processed_data (dict): Data after processing.

        Returns:
            dict: An empty dictionary or confirmation message.

        Raises:
            Exception: If the action could not be performed.
        """
        try:
            if self.action_type == 'print':
                # Example action: print the data
                print(f"Actuator action (print): {processed_data}")
                if self.logger:
                    self.logger.info(f"Actuator performed print action with data: {processed_data}")
            elif self.action_type == 'alert':
                # Example action: trigger an alert (placeholder)
                # Implement alert logic here
                if self.logger:
                    self.logger.info(f"Actuator triggered an alert with data: {processed_data}")
            else:
                if self.logger:
                    self.logger.warning(f"Unknown action type: {self.action_type}")
                raise ValueError(f"Unsupported action type: {self.action_type}")

            # Since actuators are typically the end of the pipeline, they may not send data forward
            return {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to perform action: {e}")
            raise

    def store_data(self):
        """
        Stores data if necessary.

        This method can be used to log actions taken or store results to a database.
        """
        # Implement storage logic if needed
        pass

    def on_start(self):
        """
        Actions to perform when the machine starts.

        This method is called at the beginning of the machine's lifecycle.
        """
        super().on_start()
        # Additional startup actions specific to the actuator

    def on_finish(self):
        """
        Actions to perform when the machine finishes.

        This method is called at the end of the machine's lifecycle.
        """
        super().on_finish()
        # Additional cleanup actions specific to the actuator
