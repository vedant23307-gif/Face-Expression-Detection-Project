/**
 * Main Application Orchestrator
 * Connects Webcam, MediaPipe Face Bounding Box, Deep Neural Network Classifier, and UI Controls
 */

document.addEventListener('DOMContentLoaded', async () => {
  const videoElement = document.getElementById('webcam');
  const canvasElement = document.getElementById('canvas-overlay');
  const procCanvas = document.getElementById('proc-canvas');
  const procCtx = procCanvas ? procCanvas.getContext('2d') : null;

  // UI Buttons (Optional)
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
  let fps = 30;

  let isMediaPipeActive = false;
  let isApiProcessing = false;
  let targetBox = null;
  let currentEmotion = 'Neutral';
  let currentConf = 95;

  function syncCanvasSize() {
    if (videoElement.clientWidth && videoElement.clientHeight) {
      canvasElement.width = videoElement.clientWidth;
      canvasElement.height = videoElement.clientHeight;
    } else if (videoElement.videoWidth && videoElement.videoHeight) {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
    }
    if (procCanvas && videoElement.videoWidth) {
      procCanvas.width = videoElement.videoWidth;
      procCanvas.height = videoElement.videoHeight;
    }
  }

  videoElement.addEventListener('loadedmetadata', syncCanvasSize);
  window.addEventListener('resize', syncCanvasSize);

  // Initialize MediaPipe Solution or fallback to API
  try {
    if (typeof FaceMesh !== 'undefined' && typeof Camera !== 'undefined') {
      await mpHandler.init(videoElement, canvasElement, (results) => {
        isMediaPipeActive = true;

        // Smooth Rolling 1-Second FPS Calculation
        const now = performance.now();
        frameCount++;
        const delta = now - lastFrameTime;
        if (delta >= 1000) {
          fps = Math.round((frameCount * 1000) / delta);
          frameCount = 0;
          lastFrameTime = now;
        }

        if (results.hasFace && results.features) {
          // Predict Emotion with Deep Neural Network Classifier
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

      mpHandler.start();
      isRunning = true;
    } else {
      console.warn("MediaPipe CDN unavailable. Starting high-performance API camera engine.");
      startFallbackCameraEngine();
    }
  } catch (err) {
    console.error("MediaPipe initialization error, switching to API engine:", err);
    startFallbackCameraEngine();
  }

  // Fallback Camera Loop if MediaPipe CDN is unavailable
  async function startFallbackCameraEngine() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false
      });
      videoElement.srcObject = stream;
      videoElement.onloadedmetadata = () => {
        syncCanvasSize();
        requestAnimationFrame(fallbackRenderLoop);
      };
      isRunning = true;
    } catch (e) {
      console.error("Camera access error:", e);
    }
  }

  function fallbackRenderLoop() {
    if (!isRunning || isMediaPipeActive) return;

    // Smooth 1-Second Rolling FPS
    const now = performance.now();
    frameCount++;
    const delta = now - lastFrameTime;
    if (delta >= 1000) {
      fps = Math.round((frameCount * 1000) / delta);
      frameCount = 0;
      lastFrameTime = now;
    }

    syncCanvasSize();
    const ctx = canvasElement.getContext('2d');
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    const cW = canvasElement.width;
    const cH = canvasElement.height;

    if (videoElement.videoWidth > 0 && procCtx && !isApiProcessing) {
      isApiProcessing = true;
      procCtx.drawImage(videoElement, 0, 0, procCanvas.width, procCanvas.height);
      const dataUrl = procCanvas.toDataURL('image/jpeg', 0.65);

      fetch('/api/process_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      })
      .then(r => r.json())
      .then(data => {
        if (data.has_face) {
          currentEmotion = data.dominant || 'Neutral';
          currentConf = data.confidence || 95;
          const emoji = classifier.emojis[currentEmotion] || '😐';

          charts.updateEmotionBars(data.emotions || classifier.getDefaultScores().scores);
          charts.updateMetrics(currentEmotion, currentConf, fps, null);
          charts.updateOverlay(currentEmotion, currentConf, emoji);

          if (data.box && data.box.length === 4) {
            const [rx, ry, rw, rh] = data.box;
            const scaleX = cW / procCanvas.width;
            const scaleY = cH / procCanvas.height;

            const bx = Math.round(cW - (rx + rw) * scaleX);
            const by = Math.round(ry * scaleY);
            const bw = Math.round(rw * scaleX);
            const bh = Math.round(rh * scaleY);
            targetBox = [bx, by, bw, bh];
          }
        } else {
          charts.updateMetrics('Searching Face...', 0, fps, null);
          charts.updateOverlay('Searching Face...', 0, '🔍');
          targetBox = null;
        }
      })
      .catch(e => console.error("API Sync error:", e))
      .finally(() => {
        setTimeout(() => { isApiProcessing = false; }, 120);
      });
    }

    if (targetBox) {
      mpHandler.currentEmotion = currentEmotion;
      mpHandler.currentConfidence = currentConf;
      mpHandler.canvasElement = canvasElement;
      mpHandler.ctx = ctx;
      mpHandler.drawFaceBoundingBoxFromRect(targetBox[0], targetBox[1], targetBox[2], targetBox[3]);
    }

    requestAnimationFrame(fallbackRenderLoop);
  }

  // Toggle Webcam Start / Pause
  if (btnStart) {
    btnStart.addEventListener('click', () => {
      if (!isRunning) {
        if (isMediaPipeActive) mpHandler.start();
        btnStart.innerHTML = '<i class="fas fa-pause"></i> Pause Camera';
        btnStart.classList.add('btn-active');
        isRunning = true;
      } else {
        if (isMediaPipeActive) mpHandler.stop();
        btnStart.innerHTML = '<i class="fas fa-play"></i> Start Camera';
        btnStart.classList.remove('btn-active');
        isRunning = false;
      }
    });
  }

  // Toggle Face Bounding Box
  if (btnToggleBox) {
    btnToggleBox.addEventListener('click', () => {
      mpHandler.showBoundingBox = !mpHandler.showBoundingBox;
      btnToggleBox.classList.toggle('btn-active', mpHandler.showBoundingBox);
    });
  }

  // Take Snapshot
  if (btnSnapshot) {
    btnSnapshot.addEventListener('click', () => {
      syncCanvasSize();
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
  }
});
