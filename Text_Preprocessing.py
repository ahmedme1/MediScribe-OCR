import re
from textblob import TextBlob

def clean_text(text):
    """
    Cleans the extracted text by removing special characters, fixing spaces, and correcting spellings.
    """
    # Remove unwanted special characters except alphanumeric and essential punctuation
    text = re.sub(r"[^a-zA-Z0-9.,\s]", "", text)

    # Normalize multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text

def correct_spelling(text):
    """
    Uses TextBlob to correct spelling mistakes in the text.
    """
    return str(TextBlob(text).correct())

# Sample extracted text from Tesseract OCR
ocr_text = """
6|6
| AD ve amp Lok VIO (47749)
a
/ \ . 0.50
. Pain ) tAching in adays ooeet +
{ KR-8090 spe
vg (4 6 a aant FYE HO!
"""

# Step 1: Clean text
cleaned_text = clean_text(ocr_text)
print("Cleaned Text:\n", cleaned_text)

# Step 2: Correct spelling
corrected_text = correct_spelling(cleaned_text)
print("\nCorrected Text:\n", corrected_text)
