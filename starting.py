#import numpy as np
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can change to yolov8s.pt, yolov8m.pt etc.)
model = YOLO("yolov8n.pt")

# Train the model on your custom dataset
model.train(data="unzipped_dataset/data.yaml", epochs=20, imgsz=177)
