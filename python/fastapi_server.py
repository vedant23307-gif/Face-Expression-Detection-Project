#!/usr/bin/env python3
"""
High-Performance FastAPI Facial Emotion Web Server
Fixed Tracking Engine: Zero Ghost Boxes (Instant removal when face moves) + Smooth Following.
"""

import os
import sys
import cv2
import time
import json
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

# Strict Haar Face Cascade for noise suppression
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global Telemetry State
telemetry_state = {
    "dominant": "Neutral",
    "confidence": 100,
    "emotions": {
        "happy": 0.0,
        "surprise": 0.0,
        "neutral": 1.0,
        "sad": 0.0,
        "angry": 0.0,
        "fear": 0.0,
        "disgust": 0.0
    },
    "active_faces": 0,
    "fps": 30
}

class FastFaceTracker:
    """
    Zero-Ghost Face Tracking Engine.
    Instantly erases old bounding boxes when face moves away and smoothly tracks moving face.
    """
    def __init__(self, alpha=0.50, min_face_size=110):
        self.alpha = alpha
        self.min_face_size = min_face_size
        self.tracked_faces = {}  # fid -> {box, emotion, score, all_emotions}
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
        
        # Detect faces with strict minNeighbors=7
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=7,
            minSize=(self.min_face_size, self.min_face_size)
        )

        raw_candidates = []

        for (x, y, w, h) in faces:
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < 0.60 or aspect_ratio > 1.6:
                continue

            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            emotions = fer_detector.detect_emotions(face_img)

            if emotions and len(emotions) > 0:
                emo_scores = emotions[0]["emotions"]
            else:
                emo_scores = {'happy': 0.0, 'surprise': 0.0, 'neutral': 1.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0, 'disgust': 0.0}

            dom_emotion = max(emo_scores, key=emo_scores.get).capitalize()
            score = emo_scores[max(emo_scores, key=emo_scores.get)] * 100.0

            raw_candidates.append({
                'box': (x, y, w, h),
                'emotion': dom_emotion,
                'score': score,
                'all_emotions': emo_scores
            })

        valid_candidates = self.suppress_sub_patches(raw_candidates)

        # ZERO GHOST BOXES: Create fresh active state per frame
        new_tracked = {}
        used_cand_indices = set()

        # Match new candidate boxes to existing tracked faces
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
                # Face matched! Smooth position
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

        # Add any new unassigned faces
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

        # Active output list
        active_output = []
        for fid, data in self.tracked_faces.items():
            active_output.append((data['box'], data['emotion'], data['score'], data['all_emotions']))

        return active_output

tracker = FastFaceTracker(alpha=0.50, min_face_size=110)

color_map = {
    'happy': (16, 185, 129),      # Green
    'surprise': (246, 130, 59),   # Orange
    'neutral': (139, 92, 246),   # Purple
    'sad': (100, 116, 184),      # Slate
    'angry': (239, 68, 68),       # Red
    'fear': (245, 158, 11),       # Amber
    'disgust': (20, 184, 166)     # Teal
}

def generate_video_stream():
    global telemetry_state
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error opening video stream")
        return

    prev_time = time.time()

    while True:
        success, frame = cam.read()
        if not success or frame is None:
            break

        # Flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)

        active_faces = tracker.process(frame)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        dominant = "Neutral"
        score = 100.0
        emotions_dict = {'happy': 0.0, 'surprise': 0.0, 'neutral': 1.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0, 'disgust': 0.0}

        for (box, emotion, sc, all_emotions) in active_faces:
            (x, y, w, h) = box
            color = color_map.get(emotion.lower(), (255, 255, 255))
            dominant = emotion
            score = sc
            emotions_dict = all_emotions

            # Draw Clean Face Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            # Draw Crisp Emotion Tag Badge
            badge_text = f"{emotion} ({score:.0f}%)"
            (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            badge_y1 = max(0, y - text_h - 15)
            badge_y2 = y
            cv2.rectangle(frame, (x, badge_y1), (x + text_w + 20, badge_y2), color, -1)
            cv2.putText(frame, badge_text, (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        telemetry_state["dominant"] = dominant
        telemetry_state["confidence"] = round(score)
        telemetry_state["emotions"] = emotions_dict
        telemetry_state["active_faces"] = len(active_faces)
        telemetry_state["fps"] = round(fps)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/emotion_data")
async def get_emotion_data():
    return JSONResponse(content=telemetry_state)

if __name__ == "__main__":
    import uvicorn
    print("==================================================")
    print(" 🚀 TensorFlow Web AI Dashboard Server")
    print(" Engine: Zero-Ghost Fast Face Tracker")
    print(" URL: http://localhost:8000")
    print("==================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
