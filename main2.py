import numpy as np
from ultralytics import YOLO
import joblib
import pandas as pd
import os
from datetime import datetime
import random

# Load models
plastic_model = YOLO("runs/detect/train2/weights/best.pt")
clf = joblib.load("water_quality_model.pkl")

# Load dataset
water_df = pd.read_csv("test_dataset.csv")  # Ensure this file exists

# Validate required columns
required_cols = ['Image', 'pH', 'Temperature', 'Turbidity', 'DO']
if not all(col in water_df.columns for col in required_cols):
    raise ValueError(f"Dataset must contain columns: {required_cols}")

# Initialize direction
directions = ["Straight", "Turned left", "Turned right"]
current_direction = "Straight"

# Process each row
for idx, row in water_df.iterrows():
    img_path = row['Image']

    # Predict water quality
    sample = pd.DataFrame([[row['pH'], row['Temperature'], row['Turbidity'], row['DO']]],
                          columns=['pH', 'Temperature', 'Turbidity', 'DO'])
    water_prediction = clf.predict(sample)[0]

    # Predict plastic presence
    if os.path.exists(img_path):
        results = plastic_model.predict(source=img_path, conf=0.5, show=False)
        boxes = results[0].boxes
        plastic_detected = False

        if boxes is not None and len(boxes) > 0:
            plastic_detected = any(int(box.cls.item()) in [1, 2, 3, 4] for box in boxes)  # Adjust class IDs as needed
    else:
        print(f"\nImage not found: {img_path}")
        plastic_detected = False

    # Simulate timestamp and location
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    latitude = random.randint(-90, 90)
    longitude = random.randint(-180, 180)

    # Handle directional change based on water quality
    if water_prediction == 'Impure':
        direction_change = random.choice(directions[1:])  # Turned left or right
        direction_note = f"\n*Alert - Change direction ({direction_change})*"
    else:
        current_direction = "Straight"
        direction_note = ""

    # Print formatted output
    print(f"...Entry{idx + 1}...")
    print(f"Direction: {current_direction}")
    print(f"lat, long: {latitude},{longitude}")
    print(f"Plastic detected : {'Yes' if plastic_detected else 'No'}")
    print(f"Quality: {water_prediction} {direction_note}")
    #print(f"Date & Time : {timestamp}")
