import os
from ultralytics import YOLO

# Загрузка предобученной модели
def load_model(model_path):
    model = YOLO(model_path)
    return model