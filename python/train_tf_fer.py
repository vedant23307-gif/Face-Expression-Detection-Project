#!/usr/bin/env python3
"""
TensorFlow / Keras CNN Facial Emotion Model Architecture & Training Script
Builds and trains a Keras Convolutional Neural Network for 7 emotion classification.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def build_tf_emotion_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    TensorFlow / Keras CNN Architecture
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == '__main__':
    print("==================================================")
    print(" TensorFlow / Keras Facial Emotion Model Architect")
    print(f" TensorFlow Version: {tf.__version__}")
    print("==================================================")
    
    model = build_tf_emotion_cnn()
    model.summary()
    print("Keras model compiled successfully! Save model as `emotion_tf_model.h5`.")
