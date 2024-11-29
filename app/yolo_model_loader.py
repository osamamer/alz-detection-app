# app/yolo_model.py
import torch
from PIL import Image
import io

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def predict(image_bytes):
    """Run YOLO prediction on an image file and return the result."""
    # Convert image bytes to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Run inference
    results = model(image)

    # Format results as JSON serializable data
    predictions = results.pandas().xyxy[0].to_dict(orient="records")
    return predictions
