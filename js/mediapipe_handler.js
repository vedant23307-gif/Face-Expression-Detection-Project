/**
 * Ultra-Fast 60 FPS MediaPipe Face Bounding Box & Landmark Engine
 * Aligns face bounding box perfectly over live web camera.
 */

class MediaPipeHandler {
  constructor() {
    this.faceMesh = null;
    this.camera = null;
    this.landmarks = null;
    this.showBoundingBox = true;
    this.onResultsCallback = null;
    this.currentEmotion = 'Neutral';
    this.currentEmoji = '😐';
    this.currentConfidence = 100;
  }

  async init(videoElement, canvasElement, onResultsCallback) {
    this.videoElement = videoElement;
    this.canvasElement = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.onResultsCallback = onResultsCallback;

    this.faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    this.faceMesh.setOptions({
      maxNumFaces: 4,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    this.faceMesh.onResults((results) => this.handleResults(results));

    this.camera = new Camera(this.videoElement, {
      onFrame: async () => {
        await this.faceMesh.send({ image: this.videoElement });
      },
      width: 1280,
      height: 720
    });
  }

  start() {
    if (this.camera) this.camera.start();
  }

  stop() {
    if (this.camera) this.camera.stop();
  }

  setEmotionState(dominant, emoji, confidence) {
    this.currentEmotion = dominant;
    this.currentEmoji = emoji;
    this.currentConfidence = confidence;
  }

  handleResults(results) {
    this.ctx.save();
    this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      this.landmarks = results.multiFaceLandmarks[0];

      if (this.showBoundingBox) {
        for (const faceLandmarks of results.multiFaceLandmarks) {
          this.drawFaceBoundingBox(faceLandmarks);
        }
      }

      const features = this.extractFeatureVector(this.landmarks);

      if (this.onResultsCallback) {
        this.onResultsCallback({
          landmarks: this.landmarks,
          features: features,
          faceCount: results.multiFaceLandmarks.length,
          hasFace: true
        });
      }
    } else {
      this.landmarks = null;
      if (this.onResultsCallback) {
        this.onResultsCallback({
          landmarks: null,
          features: null,
          faceCount: 0,
          hasFace: false
        });
      }
    }

    this.ctx.restore();
  }

  /**
   * Draw Clean Bounding Box perfectly centered over face
   */
  drawFaceBoundingBox(landmarks) {
    const width = this.canvasElement.width;
    const height = this.canvasElement.height;

    let minX = width, maxX = 0, minY = height, maxY = 0;

    for (const pt of landmarks) {
      // Mirror X coordinate so target box aligns over mirrored video
      const x = (1 - pt.x) * width;
      const y = pt.y * height;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const padX = (maxX - minX) * 0.12;
    const padY = (maxY - minY) * 0.16;

    const boxX = Math.max(0, minX - padX);
    const boxY = Math.max(0, minY - padY * 1.4);
    const boxW = Math.min(width - boxX, (maxX - minX) + padX * 2);
    const boxH = Math.min(height - boxY, (maxY - minY) + padY * 2.0);

    const colorMap = {
      'Happy': '#10b981',
      'Surprise': '#f97316',
      'Neutral': '#8b5cf6',
      'Sad': '#64748b',
      'Angry': '#ef4444',
      'Fear': '#f59e0b',
      'Disgust': '#14b8a6'
    };

    const color = colorMap[this.currentEmotion] || '#6366f1';

    // Draw Main Bounding Box
    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = color;
    this.ctx.strokeRect(boxX, boxY, boxW, boxH);

    // Draw Clean Emotion Tag Badge
    const badgeText = `${this.currentEmotion} (${this.currentConfidence}%)`;
    this.ctx.font = 'bold 16px Outfit, Inter, sans-serif';
    const textWidth = this.ctx.measureText(badgeText).width;
    const badgeH = 32;
    const badgeW = textWidth + 24;
    const badgeY = Math.max(10, boxY - badgeH - 4);

    // Badge Solid Background
    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.roundRect(boxX, badgeY, badgeW, badgeH, 6);
    this.ctx.fill();

    // Badge Text (White)
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText(badgeText, boxX + 12, badgeY + 22);
  }

  extractFeatureVector(landmarks) {
    const distance = (idx1, idx2) => {
      const p1 = landmarks[idx1];
      const p2 = landmarks[idx2];
      return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));
    };

    const faceH = distance(10, 152) || 1.0;
    const faceW = distance(234, 454) || 1.0;

    const leftEAR = distance(159, 145) / (distance(33, 133) || 0.001);
    const rightEAR = distance(386, 374) / (distance(362, 263) || 0.001);
    const avgEAR = (leftEAR + rightEAR) / 2.0;

    const mouthVert = distance(13, 14);
    const mouthHoriz = distance(61, 291);
    const MAR = mouthVert / (mouthHoriz || 0.001);

    const mouthCenterY = (landmarks[13].y + landmarks[14].y) / 2.0;
    const cornersY = (landmarks[61].y + landmarks[291].y) / 2.0;
    const smileIndex = (mouthCenterY - cornersY) / faceH;

    const leftBrowElev = distance(70, 159) / faceH;
    const rightBrowElev = distance(300, 386) / faceH;
    const avgBrowElev = (leftBrowElev + rightBrowElev) / 2.0;
    const innerBrowDist = distance(55, 285) / faceW;

    return {
      raw: {
        avgEAR,
        MAR,
        smileIndex,
        browElevation: avgBrowElev,
        innerBrowDist
      }
    };
  }
}
