import os
import torch
from torchvision import transforms
from PIL import Image

# ✅ Image Transformations
transform = transforms.Compose([
    transforms.Resize((32, 128)),  # Resize images to 32x128
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

# ✅ Folder Path
folder_path = "C:\\Users\\SAI\\OneDrive\\Desktop\\Chanchu\\001"

# ✅ Output File for Numeric Values
output_file = "C:\\Users\\SAI\\OneDrive\\Desktop\\Chanchu\\image_numeric_values.pt"

# ✅ Convert Images to Numeric Values
image_data = []  # List to store numeric values
image_labels = []  # List to store labels (optional, if filenames contain labels)

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("L")  # Convert to grayscale

        # Apply transformations
        image_tensor = transform(image)

        # Append to list
        image_data.append(image_tensor)

        # Optional: Extract label from filename (if applicable)
        # Assuming filenames are in the format "label_image.jpg"
        try:
            label = int(filename.split('_')[0])  # Extract label from filename
            image_labels.append(label)
        except ValueError:
            image_labels.append(-1)  # Use -1 if no label is found

# ✅ Save Numeric Values to File
image_data_tensor = torch.stack(image_data)  # Stack all tensors into a single tensor
torch.save({"images": image_data_tensor, "labels": image_labels}, output_file)

print(f"Converted {len(image_data)} images to numeric values and saved to {output_file}.")