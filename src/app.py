from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = YOLO("models/yolov8n_blood_detection.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    results = model(img_np)
    blurred_img = img_np.copy()

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
        roi = blurred_img[y1:y2, x1:x2]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
            blurred_img[y1:y2, x1:x2] = blurred_roi

    blurred_img = Image.fromarray(blurred_img)
    img_byte_arr = io.BytesIO()
    blurred_img.save(img_byte_arr, format="PNG")
    return {"filename": file.filename, "processed": True}