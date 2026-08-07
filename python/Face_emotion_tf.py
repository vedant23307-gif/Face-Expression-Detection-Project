#!/usr/bin/env python3
"""
Real-Time Multi-Face Facial Emotion Recognition System (TensorFlow / Keras Backend)
Features Non-Maximum Suppression (NMS), clean face bounding boxes, and emotion tag badges.
Clean UI: No top face-count box, clean emotion tag labels (e.g. `Neutral (97%)`).
"""

import sys

try:
    import cv2
except ImportError:
    print("❌ Error: OpenCV (`cv2`) is missing.")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("❌ Error: `tensorflow` is missing.")
    sys.exit(1)

try:
    from fer import FER
except (ImportError, AttributeError):
    try:
        from fer.fer import FER
    except ImportError:
        print("❌ Error: `fer` library missing. Run: python3 -m pip install fer")
        sys.exit(1)

def get_webcam_stream():
    """Try camera indices 0, 1, 2 for available webcam streams"""
    for idx in range(3):
        print(f"Testing camera index {idx}...")
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret and frame is not None:
                print(f"✅ Successfully connected to camera index {idx}!")
                return cam
            cam.release()
    return None

class MultiFaceTracker:
    """
    Multi-Person Face Tracking Engine with Non-Maximum Suppression (NMS).
    Filters false boxes, maintains smooth tracking, and assigns stable face states.
    """
    def __init__(self, alpha=0.35, max_missing_frames=5, min_face_size=100, min_confirm_frames=2):
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self.min_face_size = min_face_size
        self.min_confirm_frames = min_confirm_frames
        self.tracked_faces = {}  # fid -> {box, emotion, score, missing_count, confirm_count}
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
        """Non-Maximum Suppression (NMS): Removes inner sub-patches of a face"""
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

    def process(self, results, frame_shape):
        raw_candidates = []

        # 1. Dimension & Aspect Ratio Filtering
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
                'score': score
            })

        # 2. NMS Sub-Patch Suppression
        valid_candidates = self.suppress_sub_patches(raw_candidates)

        # 3. Match Candidates to Tracked Faces using IoU
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
                # Match found: Smooth position
                px, py, pw, ph = self.tracked_faces[best_id]['box']
                cx, cy, cw, ch = cand_box
                sx = int(self.alpha * cx + (1 - self.alpha) * px)
                sy = int(self.alpha * cy + (1 - self.alpha) * py)
                sw = int(self.alpha * cw + (1 - self.alpha) * pw)
                sh = int(self.alpha * ch + (1 - self.alpha) * ph)

                self.tracked_faces[best_id]['box'] = (sx, sy, sw, sh)
                self.tracked_faces[best_id]['emotion'] = cand['emotion']
                self.tracked_faces[best_id]['score'] = cand['score']
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
                    'missing_count': 0,
                    'confirm_count': 1
                }
                updated_ids.add(new_id)

        # 4. Decay or remove lost tracks
        to_delete = []
        for fid in list(self.tracked_faces.keys()):
            if fid not in updated_ids:
                self.tracked_faces[fid]['missing_count'] += 1
                if self.tracked_faces[fid]['missing_count'] > self.max_missing_frames:
                    to_delete.append(fid)

        for fid in to_delete:
            del self.tracked_faces[fid]

        # Active face list output: [(box, emotion, score)]
        active_output = []
        for fid, data in self.tracked_faces.items():
            if data['confirm_count'] >= self.min_confirm_frames and data['missing_count'] == 0:
                active_output.append((data['box'], data['emotion'], data['score']))

        return active_output

def main():
    print("==================================================")
    print(" 🎥 Real-Time Facial Emotion AI (Clean UI)")
    print(" Engine: TensorFlow / Keras")
    print(" Press 'ESC' or 'q' in the window to quit.")
    print("==================================================")

    detector = FER(mtcnn=False)
    tracker = MultiFaceTracker(alpha=0.35, max_missing_frames=5, min_face_size=100, min_confirm_frames=2)

    cam = get_webcam_stream()
    if cam is None:
        print("❌ Error: Could not access live camera.")
        return

    color_map = {
        'happy': (16, 185, 129),      # Green
        'surprise': (246, 130, 59),   # Orange
        'neutral': (139, 92, 246),   # Purple
        'sad': (100, 116, 184),      # Slate
        'angry': (239, 68, 68),       # Red
        'fear': (245, 158, 11),       # Amber
        'disgust': (20, 184, 166)     # Teal
    }

    while True:
        success, frame = cam.read()
        if not success or frame is None:
            break

        # Flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)

        # Detect emotions across all faces
        results = detector.detect_emotions(frame)

        # Process through Tracker & NMS Sub-Patch Suppressor
        active_faces = tracker.process(results, frame.shape)

        # Render bounding boxes and clean emotion tags
        for (box, emotion, score) in active_faces:
            (x, y, w, h) = box
            color = color_map.get(emotion.lower(), (255, 255, 255))

            # Draw Face Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            # Draw Clean Emotion Tag Badge (e.g. `Neutral (97%)`)
            badge_text = f"{emotion} ({score:.0f}%)"
            (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            badge_y1 = max(0, y - text_h - 15)
            badge_y2 = y
            cv2.rectangle(frame, (x, badge_y1), (x + text_w + 20, badge_y2), color, -1)
            cv2.putText(frame, badge_text, (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("TensorFlow Facial Emotion AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
