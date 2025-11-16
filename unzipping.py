import zipfile

with zipfile.ZipFile("Plastic Prediction.v7i.yolov8.zip", 'r') as zip_ref:
    zip_ref.extractall("unzipped_dataset")
