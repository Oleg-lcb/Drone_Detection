import json

import cv2 as cv
from sympy.printing.pretty.pretty_symbology import annotated

from src.utils import load_model

class DroneDetection:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    # Обнаружение на изображении
    def detect_image(self, source):
        results = self.model.predict(source)

        # Список всех детекций
        detections = []

        # Обработка данных обнаружения на одном кадре
        frame_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                # Сохранение данных обнаружения на одном кадре
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        # Вывод изображения на экран
        cv.imshow('drone', results[0].plot())
        cv.waitKey(0)
        return json.dumps(detections)



    # Обнаружения на видеофайле
    def detect_video(self, source):

        # Открытие видео
        cap = cv.VideoCapture(source)

        # Список всех детекций
        detections = []
        frame_number = 0

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Запуск отслеживания с сохранением между кадрами
                results = self.model.track(frame, persist=True)

                # Обработка данных обнаружения на одном кадре
                frame_detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])

                        # Сохранение данных обнаружения на одном кадре
                        frame_detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bounding_box": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                            }
                        })

                # Визуализация между кадрами
                annotated_frame = results[0].plot()

                # Добавление FPS в кадр
                fps = "FrameRate= " + str(cap.get(cv.CAP_PROP_FPS)) + " FPS"
                cv.putText(annotated_frame, fps, (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255), thickness=1)
                cv.imshow('Tracking', annotated_frame)

                # Добавление в общий список детекций
                for detection in frame_detections:
                    detections.append({
                        "frame_number": frame_number,
                        "detections": frame_detections
                    })

                frame_number += 1

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv.destroyAllWindows()
        return json.dumps(detections, indent=4)


