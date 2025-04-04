import easyocr

def ocr_easyocr(image_path):
    reader = easyocr.Reader(['en'])  # English Language
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

# Ensure the file path is correct
print(ocr_easyocr("Raw_Image_6.jpg"))
