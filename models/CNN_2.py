import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(57600, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 37),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
