import cv2
import pytesseract
import numpy as np
import os
from PIL import Image
from scipy.ndimage import interpolation as inter

# Step 1: Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Binarization
    img = cv2.GaussianBlur(img, (3,3), 0)  # Remove noise
    return img

# Step 2: Train Tesseract on IAM Dataset (Requires Manual Annotations)
def train_tesseract(training_data_path, output_dir):
    os.system(f"tesseract {training_data_path} {output_dir} --psm 6")

# Step 3: Implement OCR Pipeline
def perform_ocr(image_path):
    preprocessed_img = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_img, config='--psm 6')
    return text

# Step 4: Evaluate OCR Performance
def evaluate_ocr(output_text, ground_truth):
    output_words = output_text.split()
    gt_words = ground_truth.split()
    errors = sum(1 for a, b in zip(output_words, gt_words) if a != b)
    word_error_rate = errors / max(len(gt_words), 1)
    return word_error_rate

# Example Usage
if __name__ == "__main__":
    # Provide the exact path to your image file
    image_path = "C:\\Users\\SAI\\OneDrive\\Desktop\\Chanchu\\Image_4.jpg"  # Path to Image_3.jpg
    preprocessed_img = preprocess_image("C:\\Users\\SAI\\OneDrive\\Desktop\\Chanchu\\Image_4.jpg")
    cv2.imwrite("preprocessed.png", preprocessed_img)  # Save preprocessed image
    
    extracted_text = perform_ocr(image_path)
    print("Extracted Text:", extracted_text)
    
    # Example Ground Truth for evaluation (should be loaded from IAM dataset ascii files)
    ground_truth_text = "A MOVE to stop Mr. Gaitskell from nominating any more Labour life Peers..."
    wer = evaluate_ocr(extracted_text, ground_truth_text)
    print(f"Word Error Rate: {wer:.2%}")
