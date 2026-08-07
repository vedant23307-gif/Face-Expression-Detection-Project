/**
 * Main Application Orchestrator
 * Connects Webcam, MediaPipe Face Bounding Box, Deep Neural Network Classifier, and UI Controls
 */

document.addEventListener('DOMContentLoaded', async () => {
  const videoElement = document.getElementById('webcam');
  const canvasElement = document.getElementById('canvas-overlay');
  
  // UI Buttons
  const btnStart = document.getElementById('btn-start');
  const btnToggleBox = document.getElementById('btn-toggle-box');
  const btnToggleMesh = document.getElementById('btn-toggle-mesh');
  const btnSnapshot = document.getElementById('btn-snapshot');

  // Controllers
  const mpHandler = new MediaPipeHandler();
  const classifier = new EmotionClassifier();
  const charts = new ChartsController();

  let isRunning = false;
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let fps = 0;

  function fitCanvasToVideo() {
    if (videoElement.videoWidth && videoElement.videoHeight) {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
    }
  }

  videoElement.addEventListener('loadedmetadata', fitCanvasToVideo);
  window.addEventListener('resize', fitCanvasToVideo);

  // Initialize MediaPipe Solution
  await mpHandler.init(videoElement, canvasElement, (results) => {
    // FPS Calculation
    const now = performance.now();
    frameCount++;
    if (now - lastFrameTime >= 1000) {
      fps = (frameCount * 1000) / (now - lastFrameTime);
      frameCount = 0;
      lastFrameTime = now;
    }

    if (results.hasFace && results.features) {
      // Predict Emotion with 3-Layer Deep Neural Network
      const prediction = classifier.predict(results.features);

      // Pass predicted emotion to face bounding box drawer
      mpHandler.setEmotionState(prediction.dominant, prediction.emoji, prediction.confidence);

      // Update UI Metrics & Overlay
      charts.updateEmotionBars(prediction.scores);
      charts.updateMetrics(prediction.dominant, prediction.confidence, fps, results.features.raw);
      charts.updateOverlay(prediction.dominant, prediction.confidence, prediction.emoji);
    } else {
      const defaultPred = classifier.getDefaultScores();
      charts.updateEmotionBars(defaultPred.scores);
      charts.updateMetrics('No Face Detected', 0, fps, null);
      charts.updateOverlay('Searching Face...', 0, '🔍');
      mpHandler.setEmotionState('Neutral', '😐', 0);
    }
  });

  // Toggle Webcam Start / Pause
  btnStart.addEventListener('click', () => {
    if (!isRunning) {
      mpHandler.start();
      btnStart.innerHTML = '<i class="fas fa-pause"></i> Pause Camera';
      btnStart.classList.add('btn-active');
      isRunning = true;
    } else {
      mpHandler.stop();
      btnStart.innerHTML = '<i class="fas fa-play"></i> Start Camera';
      btnStart.classList.remove('btn-active');
      isRunning = false;
    }
  });

  // Toggle Face Bounding Box
  if (btnToggleBox) {
    btnToggleBox.addEventListener('click', () => {
      mpHandler.showBoundingBox = !mpHandler.showBoundingBox;
      btnToggleBox.classList.toggle('btn-active', mpHandler.showBoundingBox);
    });
  }

  // Toggle Mesh Wireframe
  if (btnToggleMesh) {
    btnToggleMesh.addEventListener('click', () => {
      mpHandler.showMesh = !mpHandler.showMesh;
      btnToggleMesh.classList.toggle('btn-active', mpHandler.showMesh);
    });
  }

  // Take Snapshot
  btnSnapshot.addEventListener('click', () => {
    fitCanvasToVideo();
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasElement.width;
    tempCanvas.height = canvasElement.height;
    const ctx = tempCanvas.getContext('2d');

    // Draw Video & Overlay
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(videoElement, -tempCanvas.width, 0, tempCanvas.width, tempCanvas.height);
    ctx.restore();

    ctx.drawImage(canvasElement, 0, 0);

    // Download snapshot image
    const link = document.createElement('a');
    link.download = `emotion-detection-snapshot-${Date.now()}.png`;
    link.href = tempCanvas.toDataURL('image/png');
    link.click();
  });

  // Auto-start Camera on page launch
  mpHandler.start();
  btnStart.innerHTML = '<i class="fas fa-pause"></i> Pause Camera';
  btnStart.classList.add('btn-active');
  isRunning = true;
});
