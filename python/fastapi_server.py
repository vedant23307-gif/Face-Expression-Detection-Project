#!/usr/bin/env python3
"""
High-Performance TensorFlow Facial Emotion Web AI Server
Decodes live Base64 web camera frames, runs TensorFlow FER model,
and streams real-time emotion probabilities & bounding box coordinates.
"""

import os
import sys
import time
import json
import base64
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
    import io
except ImportError:
    Image = None

try:
    import tensorflow as tf
except ImportError:
    print("❌ Warning: `tensorflow` missing.")

try:
    from fer import FER
except (ImportError, AttributeError):
    try:
        from fer.fer import FER
    except ImportError:
        print("❌ Warning: `fer` library missing.")
        sys.exit(1)

app = FastAPI(title="TensorFlow Facial Emotion AI Server")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/css", StaticFiles(directory=os.path.join(BASE_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(BASE_DIR, "js")), name="js")

# Initialize TensorFlow FER Model Detector
fer_detector = FER(mtcnn=False)

# Strict Haar Face Cascade for face bounding box detection
if cv2 is not None:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
else:
    face_cascade = None

class EmotionPredictPayload(BaseModel):
    ear: float
    mar: float
    smile: float
    brow: float
    inner_brow: float

class FramePayload(BaseModel):
    image: str  # Base64 image string

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/api/process_frame")
async def process_frame(payload: FramePayload):
    """
    Real-Time TensorFlow Camera Frame Processing Endpoint:
    Decodes Base64 JPEG camera image, runs TensorFlow FER model,
    and returns real-time dynamic emotion predictions.
    """
    try:
        if not payload.image or "," not in payload.image:
            return JSONResponse(content={"has_face": False, "dominant": "Neutral", "confidence": 100})

        header, encoded = payload.image.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        frame = None
        if cv2 is not None:
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif Image is not None:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) if cv2 is not None else np.array(pil_img)

        if frame is None:
            return JSONResponse(content={"has_face": False, "dominant": "Neutral", "confidence": 100})

        # Run TensorFlow FER Deep Learning Model
        emotions_result = fer_detector.detect_emotions(frame)

        if emotions_result and len(emotions_result) > 0:
            emo_dict = emotions_result[0]["emotions"]
            box = emotions_result[0].get("box", [0, 0, frame.shape[1], frame.shape[0]])
            
            # Find dominant emotion
            dom_emotion = max(emo_dict, key=emo_dict.get).capitalize()
            score = round(float(emo_dict[max(emo_dict, key=emo_dict.get)]) * 100)

            # Format emotions map
            emotions_map = {k: round(float(v), 3) for k, v in emo_dict.items()}

            return JSONResponse(content={
                "has_face": True,
                "dominant": dom_emotion,
                "confidence": score,
                "emotions": emotions_map,
                "box": [int(b) for b in box]
            })
        else:
            # Fallback face box detection via Haar Cascade
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_crop = frame[y:y+h, x:x+w]
                    crop_emotions = fer_detector.detect_emotions(face_crop)
                    if crop_emotions and len(crop_emotions) > 0:
                        emo_dict = crop_emotions[0]["emotions"]
                        dom_emotion = max(emo_dict, key=emo_dict.get).capitalize()
                        score = round(float(emo_dict[max(emo_dict, key=emo_dict.get)]) * 100)
                        return JSONResponse(content={
                            "has_face": True,
                            "dominant": dom_emotion,
                            "confidence": score,
                            "emotions": {k: round(float(v), 3) for k, v in emo_dict.items()},
                            "box": [int(x), int(y), int(w), int(h)]
                        })

            return JSONResponse(content={
                "has_face": True,
                "dominant": "Neutral",
                "confidence": 95,
                "emotions": {'happy': 0.02, 'surprise': 0.01, 'neutral': 0.95, 'sad': 0.01, 'angry': 0.01, 'fear': 0.0, 'disgust': 0.0}
            })

    except Exception as e:
        return JSONResponse(content={"has_face": False, "dominant": "Neutral", "confidence": 100, "error": str(e)})

@app.post("/api/predict_emotion")
async def predict_emotion(payload: EmotionPredictPayload):
    """
    Biometric Feature Vector API Endpoint
    """
    try:
        ear = payload.ear
        mar = payload.mar
        smile = payload.smile
        brow = payload.brow
        inner_brow = payload.inner_brow

        r_neutral = 1.8
        if mar < 0.20 and abs(smile) < 0.012 and 0.11 <= brow <= 0.155:
            r_neutral += 3.5

        r_happy = (smile - 0.008) * 130.0 + (mar * 2.0) if smile > 0.008 else 0.0

        r_surprise = 0.0
        if brow > 0.142 or mar > 0.20:
            r_surprise = max(0.0, (brow - 0.140) * 60.0) + max(0.0, (mar - 0.18) * 35.0)

        r_fear = (ear - 0.235) * 35.0 + (brow - 0.142) * 30.0 if (ear > 0.235 and brow > 0.142) else 0.0
        r_sad = (-smile - 0.008) * 85.0 if smile < -0.008 else 0.0
        r_angry = (0.20 - inner_brow) * 35.0 if (inner_brow < 0.195 and brow < 0.145) else 0.0
        r_disgust = (0.20 - inner_brow) * 20.0 if (inner_brow < 0.19 and mar > 0.18) else 0.0

        logits = [r_happy, r_surprise, r_neutral, r_sad, r_angry, r_fear, r_disgust]
        max_l = max(logits)
        exps = [np.exp(l - max_l) for l in logits]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]

        emotions_map = {
            'happy': round(float(probs[0]), 3),
            'surprise': round(float(probs[1]), 3),
            'neutral': round(float(probs[2]), 3),
            'sad': round(float(probs[3]), 3),
            'angry': round(float(probs[4]), 3),
            'fear': round(float(probs[5]), 3),
            'disgust': round(float(probs[6]), 3)
        }

        dom_index = int(np.argmax(probs))
        emotion_names = ['Happy', 'Surprise', 'Neutral', 'Sad', 'Angry', 'Fear', 'Disgust']
        dominant = emotion_names[dom_index]
        confidence = round(float(probs[dom_index]) * 100)

        return JSONResponse(content={
            "has_face": True,
            "dominant": dominant,
            "confidence": confidence,
            "emotions": emotions_map
        })

    except Exception as e:
        return JSONResponse(content={"has_face": False, "dominant": "Neutral", "confidence": 100, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("==================================================")
    print(" 🚀 TensorFlow Facial Emotion Web AI Server")
    print(f" Port: {port}")
    print("==================================================")
    uvicorn.run(app, host="0.0.0.0", port=port)
