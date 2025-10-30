# SPDX-FileCopyrightText: 2025, Roi Martínez Enríquez
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as f


class DigitFiveNet(nn.Module):
    """Compact CNN tailored for the Digit-Five benchmark."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding="same")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding="same")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding="same")
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)

        self.drop = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)

        # Inicializar pesos
        self.initialize_weights()

    def forward(self, x):
        """Compute class logits for Digit-Five inputs."""
        x1 = self.pool1(f.relu(self.conv1(x)))
        x2 = self.pool2(f.relu(self.conv2(x1)))
        x3 = self.pool3(f.relu(self.conv3(x2)))
        x4 = self.pool4(f.relu(self.conv4(x3)))

        x = torch.flatten(x4, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def initialize_weights(self):
        """Apply Kaiming initialisation to convolutional and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    """Simple convolutional autoencoder used by the FLProtector algorithm."""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Reconstruct the input image."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
