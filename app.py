# main.py or app.py

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import torch

from src.model import NeuralNetwork
from src.predict import predict_image

# ---------------------- App Setup ---------------------- #
app = FastAPI()
templates = Jinja2Templates(directory="templates")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Load Model ---------------------- #
model_path = "artifacts/model.pth"
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------------- Routes -------------------------- #
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    temp_file = "temp.jpg"
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run prediction
    label, confidence, idx = predict_image(model, temp_file, device)
    os.remove(temp_file)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": {
            "label": label,
            "confidence": f"{confidence:.2%}",
            "index": idx
        }
    })
