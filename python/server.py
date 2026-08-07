#!/usr/bin/env python3
"""
PyTorch Emotion Recognition HTTP/WebSocket Backend API Server
Provides high-accuracy real-time emotion predictions using `facial_emotion_recognition`.
"""

import sys
import base64
import json
import cv2
import numpy as np

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
    from flask import Flask, request, jsonify
    from flask_cors import CORS

try:
    from facial_emotion_recognition import EmotionRecognition
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "facial-emotion-recognition", "torch", "torchvision"])
    from facial_emotion_recognition import EmotionRecognition

app = Flask(__name__)
CORS(app)

print("Initializing PyTorch EmotionRecognition Model...")
er = EmotionRecognition(device='cpu')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        img_bytes = base64.b64decode(data['image'].split(',')[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Process frame with PyTorch EmotionRecognition model
        processed_frame = er.recognise_emotion(frame, return_type='BGR')

        # Encode processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'image': f"data:image/jpeg;base64,{encoded_img}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting PyTorch Facial Emotion API Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
