/**
 * Ultra-Fast 60 FPS Web AI Facial Expression Classifier Engine
 * Responsive Instant Updates (alpha = 0.85) & Standardized 7 Emotion Keys.
 */

class EmotionClassifier {
  constructor() {
    this.emotions = ['Happy', 'Surprise', 'Neutral', 'Sad', 'Angry', 'Fear', 'Disgust'];
    this.emojis = {
      'Happy': '😄',
      'Surprise': '😲',
      'Neutral': '😐',
      'Sad': '😢',
      'Angry': '😠',
      'Fear': '😨',
      'Disgust': '🤢'
    };

    // Exponential Moving Average (EMA) smoothing state for 60 FPS rendering
    this.smoothedScores = {
      'Happy': 0.0,
      'Surprise': 0.0,
      'Neutral': 1.0,
      'Sad': 0.0,
      'Angry': 0.0,
      'Fear': 0.0,
      'Disgust': 0.0
    };

    this.alpha = 0.85; // High responsiveness (instant frame updates)
  }

  predict(features) {
    if (!features || !features.raw) {
      return this.getDefaultScores();
    }

    const { avgEAR, MAR, smileIndex, browElevation, innerBrowDist } = features.raw;

    // Logits calculation
    let r_neutral = 1.8;
    if (MAR < 0.22 && Math.abs(smileIndex) < 0.015 && browElevation >= 0.11 && browElevation <= 0.155 && innerBrowDist >= 0.20) {
      r_neutral += 3.5;
    }

    let r_happy = (smileIndex > 0.008) ? (smileIndex - 0.008) * 120.0 + (MAR * 2.5) : 0.0;

    let r_surprised = 0.0;
    if (browElevation > 0.145 || MAR > 0.20) {
      r_surprised = Math.max(0, (browElevation - 0.142) * 50.0) + Math.max(0, (MAR - 0.18) * 25.0);
      if (MAR > 0.25) r_surprised += (MAR - 0.25) * 35.0;
    }

    let r_fearful = 0.0;
    if (avgEAR > 0.235 && browElevation > 0.142) {
      r_fearful = (avgEAR - 0.235) * 35.0 + (browElevation - 0.142) * 30.0;
      if (MAR >= 0.10 && MAR <= 0.32) r_fearful += 3.0;
      if (MAR <= 0.28 && innerBrowDist < 0.23) r_fearful *= 1.4;
    }

    let r_sad = (smileIndex < -0.008) ? (-smileIndex - 0.008) * 80.0 : 0.0;
    let r_angry = (innerBrowDist < 0.195 && browElevation < 0.145) ? (0.20 - innerBrowDist) * 30.0 + (0.15 - browElevation) * 25.0 : 0.0;
    let r_disgusted = (innerBrowDist < 0.19 && MAR > 0.18 && smileIndex < 0.0) ? (0.20 - innerBrowDist) * 20.0 + (MAR - 0.15) * 8.0 : 0.0;

    const logits = [r_happy, r_surprised, r_neutral, r_sad, r_angry, r_fearful, r_disgusted];
    const probs = this.softmax(logits);

    const resultScores = {};
    let dominantEmotion = 'Neutral';
    let maxConf = -1;

    for (let i = 0; i < this.emotions.length; i++) {
      const emotion = this.emotions[i];
      const rawProb = probs[i];

      this.smoothedScores[emotion] = (this.alpha * rawProb) + ((1 - this.alpha) * this.smoothedScores[emotion]);
      const finalScore = Math.min(100, Math.max(0, Math.round(this.smoothedScores[emotion] * 100)));
      
      resultScores[emotion] = finalScore;

      if (finalScore > maxConf) {
        maxConf = finalScore;
        dominantEmotion = emotion;
      }
    }

    return {
      scores: resultScores,
      dominant: dominantEmotion,
      confidence: maxConf,
      emoji: this.emojis[dominantEmotion]
    };
  }

  softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / (sumExps || 1));
  }

  getDefaultScores() {
    return {
      scores: {
        'Happy': 0, 'Surprise': 0, 'Neutral': 100, 'Sad': 0, 'Angry': 0, 'Fear': 0, 'Disgust': 0
      },
      dominant: 'Neutral',
      confidence: 100,
      emoji: '😐'
    };
  }
}
