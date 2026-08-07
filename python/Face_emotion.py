#!/usr/bin/env python3
"""
Real-Time Facial Emotion Recognition System (Live Webcam Feed)
Powered by PyTorch Deep Learning Model (facial_emotion_recognition) and OpenCV.
"""

import sys

# Step 1: Verify Dependencies
try:
    import cv2
except ImportError:
    print("=" * 60)
    print("❌ Error: `opencv-python` is missing in your Python environment.")
    print("👉 Please run this fix command in your terminal:")
    print("   python3 -m pip install packaging setuptools wheel opencv-python facial-emotion-recognition torch torchvision")
    print("=" * 60)
    sys.exit(1)

try:
    from facial_emotion_recognition import EmotionRecognition
except ImportError:
    print("=" * 60)
    print("❌ Error: `facial-emotion-recognition` is missing.")
    print("👉 Please run this fix command in your terminal:")
    print("   python3 -m pip install facial-emotion-recognition torch torchvision")
    print("=" * 60)
    sys.exit(1)

def main():
    print("==================================================")
    print(" 🎥 Starting Real-Time Live Webcam Facial Emotion AI")
    print(" Model: PyTorch Deep Learning Emotion Recognition")
    print(" Source: Live Camera (cv2.VideoCapture(0))")
    print(" Press 'ESC' or 'q' in the window to quit.")
    print("==================================================")

    # Initialize PyTorch Emotion Recognition Model on CPU
    er = EmotionRecognition(device='cpu')

    # Default to Live Webcam (Index 0)
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    print(f"Connecting to live camera source: {video_source}...")
    cam = cv2.VideoCapture(video_source)

    if not cam.isOpened():
        print(f"❌ Error: Could not open live camera feed ({video_source}).")
        print("Please check camera permissions in System Settings -> Privacy & Security -> Camera.")
        return

    print("✅ Live camera stream connected! Opening display window...")

    while True:
        success, frame = cam.read()
        if not success or frame is None:
            print("Failed to capture live camera frame.")
            break

        # Flip horizontally for natural mirror selfie view
        frame = cv2.flip(frame, 1)

        # Run PyTorch Deep Learning Emotion Recognizer on live frame
        frame = er.recognise_emotion(frame, return_type='BGR')

        # Display Live Camera Feed Window
        cv2.imshow("Live Camera - PyTorch Facial Emotion AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("Exiting live camera stream.")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
