# 🧠 vedant23307-gif | Face Expression Detection Project

A high-accuracy, real-time **Facial Emotion Recognition AI System** created by **vedant23307-gif**, powered by a **TensorFlow / Keras Deep Convolutional Neural Network (CNN)** backend, **FastAPI stream server**, and an **HTML5 / CSS3 / JavaScript Glassmorphism Web Interface**.

---

## ✨ Features

- **TensorFlow / Keras Deep Learning Engine**: Pre-trained CNN model detecting 7 emotions (*Happy, Surprise, Neutral, Sad, Angry, Fear, Disgust*) with high confidence and precision.
- **FastAPI Web Server (`python/fastapi_server.py`)**: Asynchronous, high-speed streaming server running on `http://localhost:8000`.
- **Zero-Ghost Fast Face Tracker**: Smooth multi-person face tracking with instant position updates and Non-Maximum Suppression (NMS) sub-patch filtering.
- **Background Noise & Blinds Suppression**: `minNeighbors=7` cascade filtering eliminates false detections on window blinds, curtains, and patterned shirts.
- **Responsive & Synchronized Web Dashboard**: HTML5, Vanilla CSS3 Glassmorphism UI, real-time emotion probability spectrum bars, and confidence metrics.

---

## 📁 Repository Structure

```text
Face Expression Detection Project/
├── index.html                   # HTML5 Web Dashboard (vedant23307-gif Header)
├── css/
│   └── style.css                # Dark-mode Glassmorphism UI Stylesheet
├── js/
│   ├── app.js                   # Application Controller & Camera Logic
│   ├── emotion_classifier.js    # Client-Side Classifier Engine
│   ├── mediapipe_handler.js     # MediaPipe Landmark & Bounding Box Renderer
│   └── charts.js                # Emotion Spectrum & Geometry Visualizer
├── python/
│   ├── fastapi_server.py        # High-Performance FastAPI TensorFlow Web Server
│   ├── Face_emotion_tf.py       # Live Desktop Webcam TensorFlow AI Script
│   └── requirements.txt         # Python Package Dependencies
└── README.md                    # Project Documentation
```

---

## 🚀 Quick Start (Local Run)

### 1. Install Dependencies
```bash
pip install -r python/requirements.txt
```

### 2. Launch FastAPI Web AI Server
```bash
python3 python/fastapi_server.py
```

### 3. Open in Browser
Navigate to **`http://localhost:8000`** in Google Chrome or any modern browser!

---

## 🖥️ Terminal Desktop Mode (Alternative)

To run directly on desktop without a web browser:
```bash
python3 python/Face_emotion_tf.py
```
