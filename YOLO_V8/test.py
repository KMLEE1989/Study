from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO('yolov8x.pt')

results = model.predict(source='0', show=True)

print(results)


# yolo v8

'''
Real time
'''


