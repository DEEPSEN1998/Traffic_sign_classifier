# src/predict.py

import torch
from PIL import Image
from src.utils import preprocess_image
from src.labels import class_labels  # ✅ Import the class_labels dictionary


def predict_image(model, image_path, device="cpu"):
    # Load and preprocess the image
    with Image.open(image_path).convert("RGB") as img:
        image_tensor = preprocess_image(img).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted.item()].item()

    # Get predicted label info
    predicted_idx = predicted.item()
    predicted_class_name = class_labels.get(predicted_idx, f"Unknown Class {predicted_idx}")

    return predicted_class_name, confidence, predicted_idx
