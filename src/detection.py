import json
import torch
import cv2 as cv

from src.utils import load_model, path_to_tensor


# Задействование cuda по доступности
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DroneDetection:
    def __init__(self, model_path):
        self.model = load_model(model_path).to(device=device)

    # Обнаружение на изображении
    def detect_image(self, source):
        source = path_to_tensor(source)
        results = self.model.predict(source)

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

        # Вывод изображения на экран
        cv.imshow('drone', results[0].plot())
        cv.waitKey(0)
        return json.dumps(frame_detections)



    # Обнаружения на видеофайле
    def detect_video(self, source):
        # Открытие видео
        cap = cv.VideoCapture(source)

        # Список всех детекций
        detections = []
        frame_number = 0

        # # Пераметры для сохранения видеофайла
        # output_path = "output.mp4"
        # fps_v = cap.get(cv.CAP_PROP_FPS)
        # frame_width = 640
        # frame_height = 480
        # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # # Создание объекта VideoWriter
        # fourcc = cv.VideoWriter_fourcc(*'mp4v')
        # out = cv.VideoWriter(output_path, fourcc, fps_v, (frame_width, frame_height))

        # if not out.isOpened():
        #     print("Ошибка при создании файла")
        #     cap.release()
        #     exit()

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame = path_to_tensor(frame)
                # Запуск отслеживания с сохранением между кадрами
                results = self.model.track(frame, persist=True, conf=0.6)

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
                        "detections": detection
                    })

                frame_number += 1
                # out.write(annotated_frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        # out.release()
        cv.destroyAllWindows()
        return json.dumps(detections, indent=4)

