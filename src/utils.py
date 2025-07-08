# src/utils.py

import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchvision import transforms
from src.labels import class_labels  # ✅ Load readable label mapping


# --- Preprocess Image for Inference ---
def preprocess_image(image: Image.Image):
    """
    Resize, normalize, and convert image to tensor.
    Output: (1, 3, 64, 64) for single image with batch dim.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),                         # Resize to match training input
        transforms.ToTensor(),                              # Convert PIL to Tensor [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],           # Normalize as in training
                             std=[0.5, 0.5, 0.5])
    ])
    image = image.convert("RGB")                            # Ensure 3 channels
    return transform(image).unsqueeze(0)                    # Add batch dimension (1, 3, 64, 64)


# --- Save Model to Path ---
def save_model(model, path="artifacts/model.pth"):
    torch.save(model.state_dict(), path)


# --- Load Model from Path ---
def load_model(model_class, path="artifacts/model.pth", device="cpu"):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# --- Visualize Batch from DataLoader ---
def visualize_batch(data_loader, n=16):
    """
    Display a batch of images with their true class labels.
    """
    images, labels = next(iter(data_loader))
    images = images[:n]
    labels = labels[:n]
    unnorm_images = images * 0.5 + 0.5  # Unnormalize back to [0, 1] range

    grid = torchvision.utils.make_grid(unnorm_images, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')

    # Create a compact label title bar
    title = " | ".join([class_labels.get(label.item(), str(label.item())) for label in labels])
    plt.title(title[:100] + "..." if len(title) > 100 else title)
    plt.show()
