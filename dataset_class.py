import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class HandwritingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom Dataset for Handwritten Text Recognition.

        Args:
        - image_paths (list): List of image file paths.
        - labels (list): Corresponding text labels.
        - transform (callable, optional): Transformations to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load Image
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        # Apply Transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 128)),  # Resize images to 32x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Example image paths and labels (replace with your actual data)
image_paths = ["C:\\Users\\SAI\\OneDrive\\Desktop\\Chanchu\\output_2.png"]
labels = ["example_label"]

# Create an instance of the dataset
dataset = HandwritingDataset(image_paths, labels, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Iterate through the DataLoader and print out some information
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print(f"Images shape: {images.shape}")
    print(f"Labels: {labels}")
    break  # Remove this break statement to iterate through the entire dataset
