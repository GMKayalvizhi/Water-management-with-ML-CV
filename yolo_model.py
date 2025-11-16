from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")                                                                 
results = model.predict(source="test7.jpeg")

img = results[0].plot()

cv2.imshow("YOLO detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()