class DetectionWindowManager:
    def __init__(self):
        self.window = []
        self.detection_flag = False

    def add_detection(self, detection):
        self.window.append(detection)

    def clear_window(self):
        self.window.clear()

    def get_detections(self):
        return self.window
