from ultralytics import YOLO

# Define dataset directory

model = YOLO(model='YOLOV8_objectDetection/yolov8n.pt',task='classify')

# paths = path.name
result = model.train(data = "/home/harsh/PROJECTS/YOLOV8_objectDetection/pokemon-dataset-1000/pokemon.yaml",epochs = 100,imgsz=640,project="YOLOV8_objectDetection")
