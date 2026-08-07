#!/usr/bin/env python3
"""
High-Performance FastAPI Facial Emotion Web Server (Cloud Compatible)
Accepts client-side webcam Base64 frames via /api/process_frame,
runs TensorFlow FER model, and returns high-accuracy predictions.
"""

import os
import sys
import cv2
import time
import json
import base64
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import tensorflow as tf
except ImportError:
    print("❌ Error: `tensorflow` missing.")
    sys.exit(1)

try:
    from fer import FER
except (ImportError, AttributeError):
    try:
        from fer.fer import FER
    except ImportError:
        print("❌ Error: `fer` library missing.")
        sys.exit(1)

app = FastAPI(title="TensorFlow Facial Emotion AI Server")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/css", StaticFiles(directory=os.path.join(BASE_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(BASE_DIR, "js")), name="js")

# Initialize TensorFlow FER Model Detector
fer_detector = FER(mtcnn=False)

# Strict Haar Face Cascade for noise & wall blinds suppression
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FramePayload(BaseModel):
    image: str  # Base64 data URL

class FastFaceTracker:
    def __init__(self, alpha=0.50, min_face_size=80):
        self.alpha = alpha
        self.min_face_size = min_face_size
        self.tracked_faces = {}
        self.next_id = 1

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        return float(interArea) / float(boxAArea + boxBArea - interArea + 1e-5)

    def suppress_sub_patches(self, candidates):
        if not candidates:
            return []

        candidates.sort(key=lambda c: c['box'][2] * c['box'][3], reverse=True)

        keep = []
        for cand in candidates:
            boxA = cand['box']
            areaA = boxA[2] * boxA[3]
            should_keep = True

            for kept_cand in keep:
                boxB = kept_cand['box']
                areaB = boxB[2] * boxB[3]

                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
                yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

                interArea = max(0, xB - xA) * max(0, yB - yA)
                if interArea > 0:
                    containment_ratio = float(interArea) / float(areaA + 1e-5)
                    iou = float(interArea) / float(areaA + areaB - interArea + 1e-5)

                    if containment_ratio > 0.35 or iou > 0.25:
                        should_keep = False
                        break

            if should_keep:
                keep.append(cand)

        return keep

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )

        raw_candidates = []

        for (x, y, w, h) in faces:
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < 0.50 or aspect_ratio > 1.8:
                continue

            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            emotions = fer_detector.detect_emotions(face_img)

            if emotions and len(emotions) > 0:
                emo_scores = emotions[0]["emotions"]
            else:
                # Full frame fallback if face crop fails
                full_emotions = fer_detector.detect_emotions(frame)
                if full_emotions and len(full_emotions) > 0:
                    emo_scores = full_emotions[0]["emotions"]
                else:
                    emo_scores = {'happy': 0.0, 'surprise': 0.0, 'neutral': 1.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0, 'disgust': 0.0}

            dom_emotion = max(emo_scores, key=emo_scores.get).capitalize()
            score = emo_scores[max(emo_scores, key=emo_scores.get)] * 100.0

            raw_candidates.append({
                'box': (int(x), int(y), int(w), int(h)),
                'emotion': dom_emotion,
                'score': float(score),
                'all_emotions': {k: float(v) for k, v in emo_scores.items()}
            })

        valid_candidates = self.suppress_sub_patches(raw_candidates)

        new_tracked = {}
        used_cand_indices = set()

        for fid, tf_data in self.tracked_faces.items():
            best_iou = 0.0
            best_idx = None
            for idx, cand in enumerate(valid_candidates):
                if idx in used_cand_indices:
                    continue
                iou = self.compute_iou(cand['box'], tf_data['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= 0.15:
                px, py, pw, ph = tf_data['box']
                cx, cy, cw, ch = valid_candidates[best_idx]['box']
                sx = int(self.alpha * cx + (1 - self.alpha) * px)
                sy = int(self.alpha * cy + (1 - self.alpha) * py)
                sw = int(self.alpha * cw + (1 - self.alpha) * pw)
                sh = int(self.alpha * ch + (1 - self.alpha) * ph)

                new_tracked[fid] = {
                    'box': (sx, sy, sw, sh),
                    'emotion': valid_candidates[best_idx]['emotion'],
                    'score': valid_candidates[best_idx]['score'],
                    'all_emotions': valid_candidates[best_idx]['all_emotions']
                }
                used_cand_indices.add(best_idx)

        for idx, cand in enumerate(valid_candidates):
            if idx not in used_cand_indices:
                new_id = self.next_id
                self.next_id += 1
                new_tracked[new_id] = {
                    'box': cand['box'],
                    'emotion': cand['emotion'],
                    'score': cand['score'],
                    'all_emotions': cand['all_emotions']
                }

        self.tracked_faces = new_tracked

        active_output = []
        for fid, data in self.tracked_faces.items():
            active_output.append((data['box'], data['emotion'], data['score'], data['all_emotions']))

        return active_output

tracker = FastFaceTracker(alpha=0.50, min_face_size=80)

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/api/process_frame")
async def process_frame(payload: FramePayload):
    """
    Cloud Web Camera API Endpoint:
    Accepts Base64 image snapshot from user's laptop browser, runs TensorFlow AI model,
    and returns bounding box coordinates + emotion probabilities.
    """
    try:
        header, encoded = payload.image.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(content={"has_face": False, "faces": []})

        active_faces = tracker.process(frame)

        face_list = []
        dominant = "Neutral"
        score = 100.0
        emotions_dict = {'happy': 0.0, 'surprise': 0.0, 'neutral': 1.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0, 'disgust': 0.0}

        for (box, emotion, sc, all_emotions) in active_faces:
            dominant = emotion
            score = sc
            emotions_dict = all_emotions
            face_list.append({
                "box": list(box),
                "emotion": emotion,
                "confidence": round(sc)
            })

        return JSONResponse(content={
            "has_face": len(face_list) > 0,
            "faces": face_list,
            "dominant": dominant,
            "confidence": round(score),
            "emotions": emotions_dict,
            "active_faces": len(face_list)
        })
    except Exception as e:
        return JSONResponse(content={"has_face": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("==================================================")
    print(" 🚀 Cloud TensorFlow Facial Emotion AI Server")
    print(f" Port: {port}")
    print("==================================================")
    uvicorn.run(app, host="0.0.0.0", port=port)
