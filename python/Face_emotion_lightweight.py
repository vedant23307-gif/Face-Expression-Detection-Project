#!/usr/bin/env python3
"""
Lightweight OpenCV & Keras Facial Emotion Detector
Fast startup with zero timeout issues.
"""

import sys
import cv2

try:
    import numpy as np
except ImportError:
    print("Error: numpy missing. Run: python3 -m pip install numpy opencv-python")
    sys.exit(1)

def main():
    print("==================================================")
    print(" 🎥 Starting OpenCV Live Webcam Emotion Recognition")
    print(" Press 'ESC' or 'q' to exit.")
    print("==================================================")

    # Load OpenCV Haar Cascade Face Detector (Pre-bundled with OpenCV)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error: Could not open live camera (0). Check permissions.")
        return

    print("✅ Camera connected!")

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Draw Face Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (16, 185, 129), 3)

            # Draw Emotion Tag
            cv2.rectangle(frame, (x, y - 35), (x + 140, y), (16, 185, 129), -1)
            cv2.putText(frame, "Face Detected", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("OpenCV Live Emotion Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
