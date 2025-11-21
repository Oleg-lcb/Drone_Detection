import cv2
from utils import load_model

class DroneDetection:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def detect_image(self, source):
        results = self.model.predict(source)
        return results[0].plot()