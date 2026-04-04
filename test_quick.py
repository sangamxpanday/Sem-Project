"""
Quick Model Test - Minimal Code
Just import model, run inference, print result
"""

from ultralytics import YOLO
from pathlib import Path
import sys

# Load model
model_path = "models/vehicle_classifier_final.pt"  # <-- Change this to your model path
if not Path(model_path).exists():
    print(f"Model not found: {model_path}")
    sys.exit(1)

model = YOLO(model_path)
print(f"✅ Model loaded")

image_path = "dataset/train/bus/0aa5df349719c704.jpeg"  # <-- Change this to your image path

# Find a test image if path doesn't exist
if not Path(image_path).exists():
    print(f"Image not found: {image_path}, searching...")
    for img in Path("dataset/val").rglob("*.jpg"):
        image_path = str(img)
        break

print(f"Testing: {image_path}")

# Run inference
results = model(image_path)
result = results[0]

# Get prediction
class_name = model.names[result.probs.top1]
confidence = result.probs.top1conf.item()

print(f"\n🎯 Prediction: {class_name}")
print(f"📊 Confidence: {confidence:.2%}")
