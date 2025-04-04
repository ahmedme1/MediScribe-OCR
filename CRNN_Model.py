import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height=32, num_classes=80, hidden_size=256):
        """
        CRNN Model for Handwritten Text Recognition.

        Args:
        - img_height (int): Height of input images.
        - num_classes (int): Number of unique characters in dataset.
        - hidden_size (int): Number of hidden units in LSTM layers.
        """
        super(CRNN, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Conv6
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # Conv7
            nn.ReLU()
        )

        # LSTM for Sequence Processing
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True),
            nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        )

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for CTC blank label

    def forward(self, x):
        """
        Forward pass of CRNN.
        """
        x = self.cnn(x)  # CNN feature extraction
        x = x.permute(0, 3, 1, 2)  # Rearrange to (batch, width, channels, height)
        x = x.squeeze(2)  # Remove height dimension

        x, _ = self.rnn(x)  # RNN for sequence modeling
        x = self.fc(x)  # Fully connected layer
        return x

# Define input shape and number of characters
img_height = 32
num_classes = 80  # Adjust based on dataset
hidden_size = 256

# Build CRNN Model
crnn_model = CRNN(img_height, num_classes, hidden_size)
print(crnn_model)
