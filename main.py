import argparse
import os.path
import cv2 as cv

from src.detection import DroneDetection
from src.utils import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DroneDetection')
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()

    # Путь к модели
    model_path = os.path.abspath('models/best.pt')
    detector = DroneDetection(model_path=model_path)

    if args.source != '0':
        # Классификация входных данных
        if args.source[len(args.source) - 3:len(args.source)] == 'jpg':
            result = detector.detect_image(source=args.source)
            print(result)

        elif args.source[len(args.source) - 3:len(args.source)] == 'mp4':
            result = detector.detect_video(source=args.source)
            print(result)



