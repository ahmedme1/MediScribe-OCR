import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

# ✅ Improved CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)  # Increased dropout for better generalization
        self.relu = nn.ReLU()

        # Calculate the input size for fc1 dynamically
        dummy_input = torch.zeros(1, 1, 32, 128)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(-1).size(0)

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 26)

    def _forward_conv(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ✅ Load Preprocessed Data from `image_numeric_values.pt`
data_file = "C:\\Users\\SAI\\OneDrive\\Desktop\\Chanchu\\image_numeric_values.pt"
data = torch.load(data_file)
images = data["images"]  # Tensor of image data
labels = torch.tensor(data["labels"])  # Tensor of labels

# ✅ Split Data into Training and Testing Sets
train_size = int(0.8 * len(images))
test_size = len(images) - train_size
train_images, test_images = torch.utils.data.random_split(images, [train_size, test_size])
train_labels, test_labels = torch.utils.data.random_split(labels, [train_size, test_size])

# ✅ Create DataLoaders
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ Initialize Model, Loss, and Optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ✅ Training Loop
num_epochs = 50  # Train for 50 epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# ✅ Accuracy Calculation
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

# ✅ Final Accuracy
accuracy = calculate_accuracy(model, test_loader)
print(f"Final Test Accuracy: {accuracy:.2f}%")
