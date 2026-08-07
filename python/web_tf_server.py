#!/usr/bin/env python3
"""
High-Performance TensorFlow Facial Emotion Web Server
Powered by TensorFlow / Keras FER Model Backend + Flask Stream Engine.
"""

import os
import sys
import cv2
import json
from flask import Flask, render_template_string, Response, jsonify

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

app = Flask(__name__, static_folder="../", static_url_path="")

# Load TensorFlow FER Model Detector
detector = FER(mtcnn=False)

class MultiFaceTracker:
    def __init__(self, alpha=0.35, max_missing_frames=5, min_face_size=100, min_confirm_frames=2):
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self.min_face_size = min_face_size
        self.min_confirm_frames = min_confirm_frames
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

                    if containment_ratio > 0.40 or iou > 0.30:
                        should_keep = False
                        break

            if should_keep:
                keep.append(cand)

        return keep

    def process(self, results):
        raw_candidates = []

        for face in results:
            (x, y, w, h) = face["box"]

            if w < self.min_face_size or h < self.min_face_size:
                continue

            aspect_ratio = float(w) / float(h)
            if aspect_ratio < 0.50 or aspect_ratio > 1.7:
                continue

            emotions = face["emotions"]
            dom_emotion = max(emotions, key=emotions.get).capitalize()
            score = emotions[max(emotions, key=emotions.get)] * 100.0

            raw_candidates.append({
                'box': (x, y, w, h),
                'emotion': dom_emotion,
                'score': score,
                'all_emotions': emotions
            })

        valid_candidates = self.suppress_sub_patches(raw_candidates)

        updated_ids = set()

        for cand in valid_candidates:
            cand_box = cand['box']
            best_iou = 0.0
            best_id = None

            for fid, tf_data in self.tracked_faces.items():
                if fid in updated_ids:
                    continue
                iou = self.compute_iou(cand_box, tf_data['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = fid

            if best_id is not None and best_iou >= 0.25:
                px, py, pw, ph = self.tracked_faces[best_id]['box']
                cx, cy, cw, ch = cand_box
                sx = int(self.alpha * cx + (1 - self.alpha) * px)
                sy = int(self.alpha * cy + (1 - self.alpha) * py)
                sw = int(self.alpha * cw + (1 - self.alpha) * pw)
                sh = int(self.alpha * ch + (1 - self.alpha) * ph)

                self.tracked_faces[best_id]['box'] = (sx, sy, sw, sh)
                self.tracked_faces[best_id]['emotion'] = cand['emotion']
                self.tracked_faces[best_id]['score'] = cand['score']
                self.tracked_faces[best_id]['all_emotions'] = cand['all_emotions']
                self.tracked_faces[best_id]['missing_count'] = 0
                self.tracked_faces[best_id]['confirm_count'] += 1
                updated_ids.add(best_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracked_faces[new_id] = {
                    'box': cand_box,
                    'emotion': cand['emotion'],
                    'score': cand['score'],
                    'all_emotions': cand['all_emotions'],
                    'missing_count': 0,
                    'confirm_count': 1
                }
                updated_ids.add(new_id)

        to_delete = [fid for fid, data in self.tracked_faces.items() 
                     if fid not in updated_ids and data['missing_count'] > self.max_missing_frames]
        for fid in to_delete:
            del self.tracked_faces[fid]

        active_output = []
        for fid, data in self.tracked_faces.items():
            if data['confirm_count'] >= self.min_confirm_frames and data['missing_count'] == 0:
                active_output.append((data['box'], data['emotion'], data['score'], data['all_emotions']))

        return active_output

tracker = MultiFaceTracker(alpha=0.35, max_missing_frames=5, min_face_size=100, min_confirm_frames=2)

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
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error opening video stream")
        return

    while True:
        success, frame = cam.read()
        if not success or frame is None:
            break

        # Flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)

        # TensorFlow Deep Learning Emotion Detection
        results = detector.detect_emotions(frame)
        active_faces = tracker.process(results)

        for (box, emotion, score, all_emotions) in active_faces:
            (x, y, w, h) = box
            color = color_map.get(emotion.lower(), (255, 255, 255))

            # Draw Face Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            # Draw Clean Emotion Tag Badge (Left-to-Right Crisp Text)
            badge_text = f"{emotion} ({score:.0f}%)"
            (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            badge_y1 = max(0, y - text_h - 15)
            badge_y2 = y
            cv2.rectangle(frame, (x, badge_y1), (x + text_w + 20, badge_y2), color, -1)
            cv2.putText(frame, badge_text, (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode Frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()

@app.route('/')
def index():
    with open(os.path.join(app.static_folder, 'index.html'), 'r') as f:
        return f.read()

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("==================================================")
    print(" 🚀 Launching TensorFlow Web AI Dashboard Server")
    print(" URL: http://localhost:8000")
    print("==================================================")
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
