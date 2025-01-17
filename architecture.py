import torch
import torch.nn as nn

class CompactCNNModel(nn.Module):
    def __init__(self, num_labels):
        super(CompactCNNModel, self).__init__()
        
        # Convolutional layers with batch normalization and max-pooling
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate the flattened size for fully connected layers
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, 3, 150, 150)
            flattened_dim = self.conv_stack(dummy_tensor).numel()

        # Fully connected layers with dropout
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.13),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x
