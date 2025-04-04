import cv2
import pytesseract
from matplotlib import pyplot as plt

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to binary (Thresholding)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Show the processed image
    plt.imshow(img, cmap='gray')
    plt.title("Preprocessed Image")
    plt.show()

    return img

def extract_text_from_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(img)

    print("Extracted Text:")
    print(text)

    return text

# Example usage
if __name__ == "__main__":
    image_path = 'Image_2.png'
    extract_text_from_image(image_path)
