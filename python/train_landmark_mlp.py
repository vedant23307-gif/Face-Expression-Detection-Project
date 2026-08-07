#!/usr/bin/env python3
"""
Deep Neural Network (MLP) Landmark Classifier Trainer
Trains a 3-Layer Deep Neural Network on 16-Dimensional Facial Biometric Vectors
and exports learned weights for JS client-side and PyTorch desktop inference.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EMOTIONS = ['Happy', 'Surprised', 'Neutral', 'Sad', 'Angry', 'Fearful', 'Disgusted']

class LandmarkEmotionMLP(nn.Module):
    """
    3-Layer Deep Neural Network Architecture for Facial Expression Classification
    """
    def __init__(self, in_features=16, num_classes=7):
        super(LandmarkEmotionMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_synthetic_landmark_dataset(n_samples_per_class=1000):
    """
    Generates synthetic landmark biometric distribution dataset modeled on FER-2013/AffectNet statistics.
    16-dim vector: [avgEAR, MAR, smileIndex, avgBrowElev, innerBrowDist, leftEAR, rightEAR,
                    mouthVertNorm, mouthHorizNorm, leftBrowSlant, rightBrowSlant, noseLipDist,
                    chinLipDist, browNoseGap, marEarRatio, smileMarProd]
    """
    np.random.seed(42)
    X = []
    y = []

    for i in range(n_samples_per_class):
        # 0: Happy
        s_index = np.random.normal(0.03, 0.012)
        mar = np.random.normal(0.20, 0.05)
        ear = np.random.normal(0.24, 0.02)
        brow_elev = np.random.normal(0.14, 0.01)
        inner_brow = np.random.normal(0.24, 0.015)
        X.append([ear, mar, s_index, brow_elev, inner_brow, ear, ear, mar*0.3, mar*0.7, 0.01, 0.01, 0.08, 0.12, 0.14, mar/(ear+0.001), s_index*mar])
        y.append(0)

        # 1: Surprised
        s_index = np.random.normal(0.002, 0.008)
        mar = np.random.normal(0.38, 0.08)
        ear = np.random.normal(0.30, 0.03)
        brow_elev = np.random.normal(0.19, 0.02)
        inner_brow = np.random.normal(0.26, 0.02)
        X.append([ear, mar, s_index, brow_elev, inner_brow, ear, ear, mar*0.4, mar*0.6, 0.02, 0.02, 0.09, 0.14, 0.17, mar/(ear+0.001), s_index*mar])
        y.append(1)

        # 2: Neutral
        s_index = np.random.normal(0.001, 0.005)
        mar = np.random.normal(0.12, 0.03)
        ear = np.random.normal(0.23, 0.015)
        brow_elev = np.random.normal(0.135, 0.01)
        inner_brow = np.random.normal(0.24, 0.01)
        X.append([ear, mar, s_index, brow_elev, inner_brow, ear, ear, mar*0.25, mar*0.6, 0.0, 0.0, 0.07, 0.11, 0.13, mar/(ear+0.001), s_index*mar])
        y.append(2)

        # 3: Sad
        s_index = np.random.normal(-0.02, 0.008)
        mar = np.random.normal(0.10, 0.03)
        ear = np.random.normal(0.19, 0.02)
        brow_elev = np.random.normal(0.13, 0.01)
        inner_brow = np.random.normal(0.23, 0.015)
        X.append([ear, mar, s_index, brow_elev, inner_brow, ear, ear, mar*0.2, mar*0.5, -0.01, -0.01, 0.07, 0.10, 0.12, mar/(ear+0.001), s_index*mar])
        y.append(3)

        # 4: Angry
        s_index = np.random.normal(-0.005, 0.006)
        mar = np.random.normal(0.12, 0.03)
        ear = np.random.normal(0.18, 0.02)
        brow_elev = np.random.normal(0.11, 0.01)
        inner_brow = np.random.normal(0.17, 0.012)
        X.append([ear, mar, s_index, brow_elev, inner_brow, ear, ear, mar*0.2, mar*0.5, -0.03, -0.03, 0.06, 0.10, 0.10, mar/(ear+0.001), s_index*mar])
        y.append(4)

        # 5: Fearful
        s_index = np.random.normal(-0.002, 0.006)
        mar = np.random.normal(0.22, 0.05)
        ear = np.random.normal(0.28, 0.025)
        brow_elev = np.random.normal(0.17, 0.015)
        inner_brow = np.random.normal(0.20, 0.012)
        X.append([ear, mar, s_index, brow_elev, inner_brow, ear, ear, mar*0.3, mar*0.6, 0.02, 0.02, 0.08, 0.12, 0.15, mar/(ear+0.001), s_index*mar])
        y.append(5)

        # 6: Disgusted
        s_index = np.random.normal(-0.01, 0.008)
        mar = np.random.normal(0.18, 0.04)
        ear = np.random.normal(0.20, 0.02)
        brow_elev = np.random.normal(0.12, 0.01)
        inner_brow = np.random.normal(0.18, 0.012)
        X.append([ear, mar, s_index, brow_elev, inner_brow, ear, ear, mar*0.25, mar*0.5, -0.02, -0.02, 0.06, 0.10, 0.11, mar/(ear+0.001), s_index*mar])
        y.append(6)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def train_and_export_weights():
    print("Generating training dataset for Deep Neural Network...")
    X, y = generate_synthetic_landmark_dataset(n_samples_per_class=1200)

    # Feature Normalization (Mean & Std)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-6
    X_norm = (X - mean) / std

    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = LandmarkEmotionMLP(in_features=16, num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    print("Training Deep Neural Network Classifier for 200 epochs...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_tensor), dim=1)
        acc = (preds == y_tensor).float().mean().item() * 100.0
        print(f"Training Complete! Final Neural Network Accuracy: {acc:.2f}%")

    # Extract Weights and Biases
    w1 = model.fc1.weight.detach().numpy().tolist() # [32, 16]
    b1 = model.fc1.bias.detach().numpy().tolist()   # [32]
    w2 = model.fc2.weight.detach().numpy().tolist() # [16, 32]
    b2 = model.fc2.bias.detach().numpy().tolist()   # [16]
    w3 = model.fc3.weight.detach().numpy().tolist() # [7, 16]
    b3 = model.fc3.bias.detach().numpy().tolist()   # [7]

    weights_dict = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'W1': w1,
        'b1': b1,
        'W2': w2,
        'b2': b2,
        'W3': w3,
        'b3': b3,
        'classes': EMOTIONS
    }

    with open('landmark_nn_weights.json', 'w') as f:
        json.dump(weights_dict, f, indent=2)

    print("Exported learned weights to `landmark_nn_weights.json`!")
    return weights_dict

if __name__ == '__main__':
    train_and_export_weights()
