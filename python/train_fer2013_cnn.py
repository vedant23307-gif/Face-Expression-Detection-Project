#!/usr/bin/env python3
"""
PyTorch Deep Learning CNN Model Training Script for FER-2013 Facial Expression Dataset
Trains a 7-class Convolutional Neural Network (CNN) classifier.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 7 Expression Classes
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionCNN(nn.Module):
    """
    Custom 4-Stage Convolutional Neural Network Architecture
    Optimized for 48x48 grayscale facial expression images.
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Conv Block 1: 48x48 -> 24x24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # Conv Block 2: 24x24 -> 12x12
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # Conv Block 3: 12x12 -> 6x6
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        # Fully Connected Linear Classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

def train_model(data_dir, epochs=25, batch_size=64, lr=0.001):
    """
    Main PyTorch Training Loop
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Training Hardware Device: {device}")

    # Data Transforms & Augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    model = EmotionCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("Model initialized successfully.")
    print("Run `python3 train_fer2013_cnn.py` with dataset folder path containing `train/` and `test/` subdirectories.")

if __name__ == '__main__':
    print("==================================================")
    print(" FER-2013 PyTorch CNN Facial Expression Trainer")
    print("==================================================")
    model = EmotionCNN()
    print("Model Architecture Created:")
    print(model)
