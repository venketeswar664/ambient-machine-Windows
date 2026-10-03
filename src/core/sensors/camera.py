# src/core/sensors/camera.py

from src.core.machine import Machine
import threading
import time
from src.core.sensors.ip_camera import IP_Camera  # Adjust the import path if necessary
from src.database.database import Database

class Camera(Machine):
    """
    Camera machine that initializes an IP_Camera instance and starts reading frames.
    """

    def __init__(self, machine_name, state_manager):
        super().__init__(machine_name, state_manager)
        self.database = Database()
        self.database.connect_db()
        self.ip_camera = None

    def process(self, input_data):
        """
        Initializes the IP_Camera instance and starts frame reading.
        """
        camera_doc = input_data.get('active_camera', {})
        if not camera_doc:
            if self.logger:
                self.logger.error("No 'active_camera' provided.")
            return {}

        # Extract camera info from camera_doc
        ip_address = camera_doc.get('cameraAddress')
        device_name = camera_doc.get('deviceName', 'camera-001')
        zones = camera_doc.get('zones', [])
        process_skip_frame = camera_doc.get('process_skip_frame', 1)
        rotation = camera_doc.get('rotation' , 0)

        # Create IP_Camera instance
        self.ip_camera = IP_Camera(ip_address, device_name, zones, process_skip_frame , rotation=rotation)
        self.ip_camera.connect()

        # Start reading frames in a separate thread
        self.ip_camera.is_open = True
        thread = threading.Thread(target=self.ip_camera.start_stream_read)
        thread.start()

        # Optionally, wait to ensure frames are being read
        time.sleep(3)

        return {'ip_camera': self.ip_camera}

    def send_data(self, processed_data):
        """
        Sends the IP_Camera instance to the shared data_dict.
        """
        super().send_data(processed_data)

    def on_start(self):
        super().on_start()
        if self.logger:
            self.logger.info("Camera machine started.")

    def on_finish(self):
        # Stop the IP_Camera instance if needed
        # if self.ip_camera:
        #     self.ip_camera.is_open = False
        super().on_finish()