# sensor_machine.py

from machine import Machine

class SensorMachine(Machine):
    """
    A machine that collects data from sensors.

    Responsible for interfacing with sensor hardware or APIs to collect data
    and pass it to the pipeline for processing.
    """

    def __init__(self, config, logger=None):
        """
        Initializes the SensorMachine with configuration and logger.

        Args:
            config (dict): Configuration parameters for the sensor.
            logger (Logger, optional): Logger instance for logging events. Defaults to None.
        """
        super().__init__(config, logger)
        # Initialize sensor-specific attributes here

    def get_data(self, input_data):
        """
        Retrieves data from the sensor.

        Args:
            input_data (dict): Input data required by the sensor (usually empty).

        Returns:
            dict: Data collected from the sensor.
        """
        # Implement sensor data collection logic
        data = {}  # Replace with actual data collection
        if self.logger:
            self.logger.debug("Sensor data collected.")
        return data

    def process(self, data):
        """
        Processes the sensor data.

        Args:
            data (dict): Raw data collected from the sensor.

        Returns:
            dict: Processed data ready for the next module.

        Raises:
            ValueError: If the input data is invalid.
        """
        if not data:
            raise ValueError("No data received from the sensor.")
        # Implement data processing logic if needed
        processed_data = data  # Modify as necessary
        if self.logger:
            self.logger.debug("Sensor data processed.")
        return processed_data

    def send_data(self, processed_data):
        """
        Sends the processed data to the next module.

        Args:
            processed_data (dict): Data after processing.

        Returns:
            dict: The data to be passed along in the pipeline.
        """
        self.output_data = processed_data
        if self.logger:
            self.logger.info(f"Sending data: {processed_data}")
        return processed_data

    def store_data(self):
        """
        Stores data if necessary.

        This method can be used to store data to a database or file system.
        """
        # Implement storage logic if needed
        pass

    def on_start(self):
        """
        Actions to perform when the machine starts.

        This method is called at the beginning of the machine's lifecycle.
        """
        super().on_start()
        # Additional startup actions specific to the sensor

    def on_finish(self):
        """
        Actions to perform when the machine finishes.

        This method is called at the end of the machine's lifecycle.
        """
        super().on_finish()
        # Additional cleanup actions specific to the sensor
