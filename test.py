import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import torch

model = YOLO("/home/harsh/PROJECTS/YOLOV8_objectDetection/runs/detect/train/weights/best.pt")
path ="ash-s-pokemon-returns-ash-infernape-gary-vs-moltres-pokemon-journeys-episode-68-amv-1080-publer.io.mp4"

# This shows the result....
results = model.predict(source='YOLOV8_objectDetection/download.mp4', show=True,stream=True)
ll = list(results)
print(ll)
