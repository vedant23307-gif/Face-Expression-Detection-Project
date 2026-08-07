/**
 * UI Metrics & Emotion Progress Bars Controller
 * Connects emotion prediction scores directly to visual progress bars.
 */

class ChartsController {
  constructor() {
    this.history = [];
  }

  updateEmotionBars(scores) {
    if (!scores) return;

    for (const [emotion, score] of Object.entries(scores)) {
      const lowerName = emotion.toLowerCase();
      const fillElement = document.getElementById(`fill-${lowerName}`);
      const valElement = document.getElementById(`val-${lowerName}`);

      if (fillElement) {
        fillElement.style.width = `${score}%`;
      }
      if (valElement) {
        valElement.textContent = `${score}%`;
      }
    }
  }

  updateMetrics(dominant, confidence, fps, geometry) {
    const domElement = document.getElementById('metric-dominant');
    if (domElement) {
      domElement.textContent = dominant;
    }

    const confElement = document.getElementById('metric-confidence');
    if (confElement) {
      confElement.textContent = `${confidence}%`;
    }

    const fpsElement = document.getElementById('metric-fps');
    if (fpsElement) {
      fpsElement.textContent = Math.round(fps);
    }

    if (geometry) {
      const earEl = document.getElementById('geo-ear');
      const marEl = document.getElementById('geo-mar');
      const smileEl = document.getElementById('geo-smile');
      const browEl = document.getElementById('geo-brow');

      if (earEl) earEl.textContent = geometry.avgEAR.toFixed(3);
      if (marEl) marEl.textContent = geometry.MAR.toFixed(3);
      if (smileEl) smileEl.textContent = geometry.smileIndex.toFixed(4);
      if (browEl) browEl.textContent = geometry.browElevation.toFixed(3);
    }
  }

  updateOverlay(dominant, confidence, emoji) {
    const emojiEl = document.getElementById('overlay-emoji');
    const labelEl = document.getElementById('overlay-label');
    const confEl = document.getElementById('overlay-conf');

    if (emojiEl) emojiEl.textContent = emoji;
    if (labelEl) labelEl.textContent = dominant;
    if (confEl) confEl.textContent = `${confidence}% match`;
  }
}
