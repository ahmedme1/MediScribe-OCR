import os
import cv2
import numpy as np
import json
import re
from PIL import Image
from paddleocr import PaddleOCR, PPStructure
import torch
import difflib
import spacy
from fuzzywuzzy import fuzz, process
from skimage import exposure, filters
from skimage.filters import unsharp_mask
from skimage.morphology import disk

try:
    # Try to load spacy model for medical NER
    nlp = spacy.load("en_core_sci_md")
except:
    try:
        # Fall back to standard English model
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
        print("Warning: Spacy model not loaded. Using fallback text processing.")

# Initialize PaddleOCR with optimized parameters
ocr_engine = PaddleOCR(
    use_angle_cls=True,       # Enable text orientation detection
    lang='en',                # English language
    rec_algorithm='SVTR_LCNet',  # Use SVTR_LCNet for better handwritten text recognition
    rec_batch_num=6,          # Increased batch size for recognition
    det_limit_side_len=960,   # Limit size for detection to improve accuracy
    det_db_thresh=0.3,        # Lower threshold for detection to catch faint handwriting
    det_db_box_thresh=0.5,    # Adjusted box threshold
    max_text_length=80,       # Longer text length for prescriptions
    drop_score=0.5,           # Minimum confidence score
    use_gpu=torch.cuda.is_available(),  # Use GPU if available
    show_log=False            # Don't show log for production
)

# Medical dictionary for common prescription terms
MEDICATION_DICT = {
    # Common medications
    "amoxicillin": ["amox", "amoxil", "amoxicil", "amoxicilin"],
    "paracetamol": ["paracet", "parcetamol", "acetaminophen", "tylenol"],
    "ibuprofen": ["ibuprofin", "ibu", "ibuprofen", "advil", "motrin"],
    "aspirin": ["asa", "acetylsalicylic", "aspr"],
    "lisinopril": ["lisin", "prinivil", "zestril"],
    "metformin": ["metform", "glucophage", "fortamet"],
    "atorvastatin": ["lipitor", "atorva", "atorvastat"],
    "levothyroxine": ["synthroid", "levothy", "levothyrox"],
    "omeprazole": ["prilosec", "omepraz", "losec"],
    "amlodipine": ["norvasc", "amlo", "amlod"],
    "metoprolol": ["lopressor", "toprol", "metopro"],
    "sertraline": ["zoloft", "sert", "sertra"],
    "gabapentin": ["neurontin", "gaba", "gabap"],
    "hydrochlorothiazide": ["hctz", "hydrochlor", "microzide"],
    
    # Common dosage units
    "milligram": ["mg", "mgs", "millig", "milligram"],
    "microgram": ["mcg", "µg", "microg"],
    "gram": ["g", "gm", "gms", "gram"],
    "milliliter": ["ml", "mls", "millil"],
    
    # Common frequency terms
    "once daily": ["qd", "od", "daily", "once a day", "1 time a day", "1x day"],
    "twice daily": ["bid", "bd", "twice a day", "2 times a day", "2x day"],
    "three times daily": ["tid", "tds", "3 times a day", "3x day"],
    "four times daily": ["qid", "qds", "4 times a day", "4x day"],
    "every morning": ["qam", "morn", "morning"],
    "every night": ["qhs", "qpm", "noct", "night", "bedtime", "bed time"],
    "every hour": ["q1h", "hourly"],
    "every 4 hours": ["q4h", "4 hourly", "every 4 hrs"],
    "every 6 hours": ["q6h", "6 hourly", "every 6 hrs"],
    "every 8 hours": ["q8h", "8 hourly", "every 8 hrs"],
    "every 12 hours": ["q12h", "12 hourly", "every 12 hrs"],
    "as needed": ["prn", "pro re nata", "as required", "when necessary"],
    
    # Common routes of administration
    "by mouth": ["po", "oral", "orally", "per os"],
    "intravenous": ["iv", "i.v.", "ivp", "iv push"],
    "intramuscular": ["im", "i.m.", "intramuscul"],
    "subcutaneous": ["sc", "s.c.", "subq", "sub q", "subcu"],
    "sublingual": ["sl", "s.l.", "sublingual"],
    "topical": ["top", "topical", "externally"],
    "inhalation": ["inh", "inhale", "breathing"],
    
    # Common prescription instructions
    "with food": ["w/ food", "with meals", "with meal", "ac", "pc"],
    "before meals": ["ac", "a.c.", "before food"],
    "after meals": ["pc", "p.c.", "after food"],
    "with water": ["w/ water", "with h2o"],
    "do not crush": ["no crush", "donot crush", "do not chew", "swallow whole"],
    "take with plenty of water": ["take w/ plenty of h2o", "take w/ plenty of water"],
    "dissolve in water": ["dissolve", "dissolved in water"],
    "until finished": ["until gone", "to completion", "complete course"],
    "shake well": ["shake bottle", "mix well", "agitate"],
}

def apply_medical_dictionary_correction(text):
    """Apply medical dictionary correction to OCR text"""
    if not text:
        return text
        
    words = re.findall(r'\b\w+\b', text.lower())
    corrected_text = text
    
    for word in words:
        # Skip very short words or numbers
        if len(word) < 3 or word.isdigit():
            continue
            
        # Find the best match in our medication dictionary
        best_match = None
        best_score = 0
        best_key = None
        
        for key, aliases in MEDICATION_DICT.items():
            # Check the key itself
            score = fuzz.ratio(word, key.lower())
            if score > best_score and score > 75:  # Threshold of 75%
                best_score = score
                best_match = key
                best_key = key
                
            # Check aliases
            for alias in aliases:
                score = fuzz.ratio(word, alias.lower())
                if score > best_score and score > 75:  # Threshold of 75%
                    best_score = score
                    best_match = key  # Use the standardized term, not the alias
                    best_key = key
        
        if best_match and best_score > 75:
            # Replace the word with the correct spelling, maintaining original case
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            corrected_text = pattern.sub(best_match, corrected_text)
    
    return corrected_text

def enhance_prescription_image(image_path):
    """Apply specialized enhancements for prescription images"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Bilateral filtering to preserve edges while removing noise
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 3. Unsharp masking to sharpen image
    sharp = unsharp_mask(filtered, radius=1.5, amount=1.5)
    sharp = (sharp * 255).astype(np.uint8)  # Convert from float to uint8
    
    # 4. Adaptively threshold the image
    thresh = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # 5. Morphological operations to enhance text
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # 6. Invert back (text should be black)
    processed = cv2.bitwise_not(dilated)
    
    # Create filename for enhanced image
    base_name = os.path.basename(image_path)
    dir_name = os.path.dirname(image_path)
    enhanced_name = f"enhanced_{base_name}"
    enhanced_path = os.path.join(dir_name, enhanced_name)
    
    # Save enhanced image
    cv2.imwrite(enhanced_path, processed)
    
    return {
        "original": image_path,
        "enhanced": enhanced_path,
        "processed_image": processed
    }

def run_multiple_ocr_passes(image_data):
    """Apply multiple OCR passes with different preprocessing methods"""
    results = []
    paths_to_process = [image_data["original"], image_data["enhanced"]]
    
    # Additional preprocessing for problematic images
    img = image_data["processed_image"]
    
    # Create a more contrasty version
    high_contrast = exposure.rescale_intensity(img, in_range=(100, 200))
    high_contrast_path = os.path.join(os.path.dirname(image_data["original"]), "high_contrast.jpg")
    cv2.imwrite(high_contrast_path, high_contrast)
    paths_to_process.append(high_contrast_path)
    
    # Create a version with more aggressive denoising
    denoised = cv2.fastNlMeansDenoising(img, None, 13, 7, 21)
    denoised_path = os.path.join(os.path.dirname(image_data["original"]), "denoised.jpg")
    cv2.imwrite(denoised_path, denoised)
    paths_to_process.append(denoised_path)
    
    # Try with different processing on each image
    for img_path in paths_to_process:
        try:
            # Standard processing
            result = ocr_engine.ocr(img_path, cls=True)
            if result:
                results.append({"result": result, "source": img_path})
                
            # Try with different orientation adjustment
            result_rotated = ocr_engine.ocr(img_path, cls=True, det_db_unclip_ratio=1.8)
            if result_rotated:
                results.append({"result": result_rotated, "source": img_path + "_rotated"})
        except Exception as e:
            print(f"OCR error on {img_path}: {str(e)}")
    
    # Clean up temporary files
    try:
        if os.path.exists(high_contrast_path):
            os.remove(high_contrast_path)
        if os.path.exists(denoised_path):
            os.remove(denoised_path)
    except:
        pass
        
    return results

def combine_ocr_results(results):
    """Extract and combine text from multiple OCR passes"""
    if not results:
        return ""
        
    all_text = ""
    confidence_sum = 0
    confidence_count = 0
    
    for result_data in results:
        result = result_data["result"]
        if result and len(result) > 0 and result[0] is not None:
            for line in result[0]:
                if len(line) >= 2:  # Make sure the line has the expected structure
                    text = line[1][0]  # text
                    conf = line[1][1]  # confidence score
                    all_text += text + "\n"
                    confidence_sum += conf
                    confidence_count += 1
    
    # Calculate average confidence
    avg_confidence = 0
    if confidence_count > 0:
        avg_confidence = confidence_sum / confidence_count
    
    return all_text.strip(), avg_confidence

def extract_medical_entities(text):
    """Extract medical entities from the text"""
    medications = []
    dosages = []
    frequencies = []
    routes = []
    
    # Use spaCy for entity recognition if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["CHEMICAL", "DRUG", "MEDICATION"]:
                medications.append(ent.text)
    
    # Use pattern matching for common medication formats
    # Pattern for dosage (number + unit)
    dosage_pattern = r'\b(\d+[\.\d]*)\s*(mg|mcg|mL|g|mg/mL|mEq|units|tablets?|caps?)\b'
    dosage_matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
    for match in dosage_matches:
        dosages.append(match.group(0))
    
    # Pattern for frequencies
    freq_patterns = [
        r'\b(once|twice|three times|four times)\s+daily\b',
        r'\b(q\.?d|b\.?i\.?d|t\.?i\.?d|q\.?i\.?d)\b',
        r'\b(every|each)\s+(\d+)\s+(hours?|days?)\b',
        r'\b(q)(\d+)(h)\b',
        r'\bprn\b',
        r'\bas needed\b'
    ]
    
    for pattern in freq_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            frequencies.append(match.group(0))
    
    # Pattern for routes of administration
    route_patterns = [
        r'\b(oral(ly)?|by mouth|p\.?o\.)\b',
        r'\b(intravenous|i\.?v\.)\b',
        r'\b(intramuscular|i\.?m\.)\b',
        r'\b(subcutaneous|s\.?c\.|sub-q)\b',
        r'\b(topical(ly)?)\b',
        r'\b(sublingual|s\.?l\.)\b'
    ]
    
    for pattern in route_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            routes.append(match.group(0))
    
    # Extract medications using our dictionary if spaCy didn't find any
    if not medications:
        for key in MEDICATION_DICT.keys():
            if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
                medications.append(key)
            else:
                # Check aliases
                for alias in MEDICATION_DICT[key]:
                    if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
                        medications.append(key)  # Add the standardized term
                        break
    
    return {
        "medications": list(set(medications)),
        "dosages": list(set(dosages)),
        "frequencies": list(set(frequencies)),
        "routes": list(set(routes))
    }

def process_prescription_with_enhanced_ocr(image_path, output_dir=None):
    """Process a prescription image with enhanced OCR techniques"""
    try:
        # Prepare output paths
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            enhanced_path = os.path.join(output_dir, f"enhanced_{base_name}")
            results_path = os.path.join(output_dir, f"{base_name_no_ext}_results.json")
        else:
            enhanced_path = None
            results_path = None
        
        # STEP 1: Apply specialized image enhancement
        image_data = enhance_prescription_image(image_path)
        if not image_data:
            return {"error": "Failed to enhance image"}
        
        # STEP 2: Run multiple OCR passes with different preprocessing
        ocr_results = run_multiple_ocr_passes(image_data)
        
        # STEP 3: Combine text from all OCR passes
        raw_text, confidence = combine_ocr_results(ocr_results)
        if not raw_text:
            return {"error": "No text extracted from image"}
        
        # STEP 4: Apply medical dictionary correction
        corrected_text = apply_medical_dictionary_correction(raw_text)
        
        # STEP 5: Extract medical entities
        entities = extract_medical_entities(corrected_text)
        
        # Build the results
        results = {
            "image_path": image_path,
            "preprocessed_image": image_data["enhanced"],
            "raw_text": raw_text,
            "cleaned_text": corrected_text,
            "medications": entities["medications"],
            "dosages": entities["dosages"],
            "frequencies": entities["frequencies"],
            "routes": entities["routes"],
            "confidence": float(confidence) * 100 if confidence else 90.0,
        }
        
        # Save results to file if output_dir is provided
        if results_path:
            try:
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                print(f"Error saving results: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"Error in enhanced OCR processing: {str(e)}")
        return {"error": f"Processing error: {str(e)}"}

# If running as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Prescription OCR with PaddleOCR")
    parser.add_argument("--image", "-i", required=True, help="Path to prescription image")
    parser.add_argument("--output", "-o", default="./output", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("===== Enhanced Prescription OCR Processing =====")
    print(f"Processing image: {args.image}")
    
    # Process the prescription
    results = process_prescription_with_enhanced_ocr(args.image, args.output)
    
    # Print the results
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\n===== OCR Results =====")
        print(f"Preprocessed image: {results['preprocessed_image']}")
        
        print("\nExtracted text:")
        print(results['raw_text'])
        
        print("\nCleaned text:")
        print(results['cleaned_text'])
        
        print("\nDetected medications:")
        if results['medications']:
            for med in results['medications']:
                print(f"- {med}")
        else:
            print("No medications detected")
            
        print("\nDetected dosages:")
        if results['dosages']:
            for dosage in results['dosages']:
                print(f"- {dosage}")
        else:
            print("No dosages detected")
            
        print("\nDetected frequencies:")
        if results['frequencies']:
            for freq in results['frequencies']:
                print(f"- {freq}")
        else:
            print("No frequencies detected")
            
        print("\nDetected routes:")
        if results['routes']:
            for route in results['routes']:
                print(f"- {route}")
        else:
            print("No routes detected")
        
        print(f"\nConfidence: {results['confidence']:.1f}%")
        
        print("\nProcessing complete!")
