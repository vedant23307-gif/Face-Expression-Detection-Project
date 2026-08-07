#!/usr/bin/env python3
"""
OpenCV Desktop Facial Expression Detection Script
Powered by PyTorch Deep Learning `facial_emotion_recognition` model.
Draws bounding boxes around detected faces with high-accuracy emotion labels.
"""

import sys
import cv2

try:
    from facial_emotion_recognition import EmotionRecognition
except ImportError:
    import subprocess
    print("Installing `facial-emotion-recognition` package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "facial-emotion-recognition", "torch", "torchvision"])
    from facial_emotion_recognition import EmotionRecognition

def main():
    print("==================================================")
    print(" Starting High-Accuracy PyTorch Facial Emotion AI")
    print(" Powered by pre-trained EmotionRecognition Model")
    print(" Press 'ESC' or 'q' to exit.")
    print("==================================================")

    # Initialize pre-trained PyTorch Emotion Classifier
    er = EmotionRecognition(device='cpu')

    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    cam = cv2.VideoCapture(video_source)
    if not cam.isOpened():
        print(f"Error: Could not open camera or video stream {video_source}")
        return

    while True:
        success, frame = cam.read()
        if not success or frame is None:
            print("End of video stream or failed to read frame.")
            break

        # Process frame with PyTorch deep learning model
        frame = er.recognise_emotion(frame, return_type='BGR')

        # Display Real-time Window
        cv2.imshow("PyTorch Facial Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
