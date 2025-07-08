

---

````markdown
# 🚦 Traffic Sign Classifier (FastAPI + PyTorch + Docker)

This project is a deep learning-based Traffic Sign Recognition system using a Convolutional Neural Network (CNN) trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news). It allows users to upload an image of a traffic sign and returns the predicted class using a FastAPI web interface.

---

## 📌 Project Features

- ✅ CNN model trained on GTSRB with **93% accuracy**
- ✅ **FastAPI** backend for serving the model
- ✅ **Dockerized** for deployment
- ✅ Ready to deploy on platforms like **Render**
- ✅ Supports image upload and returns predictions in real time

---

## 🧠 Model Architecture

- Input: 64x64 RGB images  
- 3 Convolutional Layers + MaxPooling  
- 2 Fully Connected Layers  
- Trained with CrossEntropyLoss and Adam Optimizer  
- Achieved ~93% accuracy on the test set  

---

## 📁 Project Structure

```bash
traffic-sign-classifier/
├── app/
│   ├── main.py               # FastAPI app
│   ├── templates/            # Jinja2 templates (HTML frontend)
│   │   └── index.html
│   └── static/               # Optional: CSS, JS, or uploaded images
│
├── src/
│   ├── model.py              # CNN model architecture
│   ├── predict.py            # Inference function
│   └── labels.py             # Class label dictionary (0–42)
│
├── saved_models/
│   └── model.pth             # Trained model weights
│
├── Dockerfile
├── requirements.txt
└── README.md
````

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/traffic-sign-classifier.git
cd traffic-sign-classifier
```

### 2. Create Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the App Locally with FastAPI

```bash
uvicorn app.main:app --reload
```

Open your browser at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🐳 Run with Docker

### 1. Build Docker Image

```bash
docker build -t traffic-sign-classifier .
```

### 2. Run Docker Container

```bash
docker run -d -p 8000:8000 traffic-sign-classifier
```

Visit the app at: [http://localhost:8000](http://localhost:8000)

---

## 🌐 Deployment on Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Set the **Start Command** to:

```bash
gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

4. Configure environment:

   * Runtime: Python 3
   * Port: `8000` or `$PORT`

---

## 🏷️ Class Labels

There are 43 traffic sign classes (0 to 42). Examples include:

* `0`: Speed limit (20km/h)
* `1`: Speed limit (30km/h)
* ...
* `14`: Stop
* `17`: No entry
* ...
* `42`: End of no passing by vehicles over 3.5 metric tons

Complete mapping is available in `src/labels.py`.

---




