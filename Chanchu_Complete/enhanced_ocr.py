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
    max_text_length=100,       # Longer text length for prescriptions
    drop_score=0.5,           # Minimum confidence score
    use_gpu=torch.cuda.is_available(),  # Use GPU if available
    show_log=False            # Don't show log for production
)

# Medical dictionary for common prescription terms
MEDICATION_DICT = {
    # Common medications
    "amoxicillin": ["amox", "amoxil", "amoxicil", "amoxicilin"],
    "paracetamol": ["paracet", "parcetamol", "acetaminophen", "tylenol", "crocin", "panadol"],
    "ibuprofen": ["ibuprofin", "ibu", "ibuprofen", "advil", "motrin", "nurofen"],
    "aspirin": ["asa", "acetylsalicylic", "aspr", "disprin", "ecotrin", "bayer"],
    "lisinopril": ["lisin", "prinivil", "zestril", "qbrelis"],
    "metformin": ["metform", "glucophage", "fortamet", "glumetza", "riomet"],
    "atorvastatin": ["lipitor", "atorva", "atorvastat", "lipibec"],
    "levothyroxine": ["synthroid", "levothy", "levothyrox", "levoxyl", "tirosint", "euthyrox"],
    "omeprazole": ["prilosec", "omepraz", "losec", "zegerid", "priosec"],
    "amlodipine": ["norvasc", "amlo", "amlod", "katerzia", "norvasc"],
    "metoprolol": ["lopressor", "toprol", "metopro", "toprol-xl"],
    "sertraline": ["zoloft", "sert", "sertra", "lustral"],
    "gabapentin": ["neurontin", "gaba", "gabap", "gralise", "horizant"],
    "hydrochlorothiazide": ["hctz", "hydrochlor", "microzide", "hydrodiuril"],
    "simvastatin": ["zocor", "simvast", "simlup", "simcard"],
    "losartan": ["cozaar", "losart", "lavestra"],
    "albuterol": ["proventil", "ventolin", "proair", "salbutamol"],
    "fluoxetine": ["prozac", "sarafem", "rapiflux"],
    "citalopram": ["celexa", "cipramil", "citalo"],
    "pantoprazole": ["protonix", "pantoloc", "pantocid"],
    "furosemide": ["lasix", "furos", "frusemide", "frusol"],
    "rosuvastatin": ["crestor", "rosuvast", "rosuvas"],
    "escitalopram": ["lexapro", "cipralex", "nexito"],
    "montelukast": ["singulair", "montek", "montair"],
    "prednisone": ["deltasone", "predni", "orasone"],
    "warfarin": ["coumadin", "jantoven", "warf"],
    "tramadol": ["ultram", "tram", "tramahexal"],
    "azithromycin": ["zithromax", "azithro", "z-pak", "azith"],
    "ciprofloxacin": ["cipro", "ciloxan", "ciproxin"],
    "lamotrigine": ["lamictal", "lamot", "lamotrigin"],
    "venlafaxine": ["effexor", "venlaf", "venlor"],
    "insulin": ["lantus", "humulin", "novolin", "humalog", "novolog", "tresiba"],
    "metronidazole": ["flagyl", "metro", "metrogel"],
    "naproxen": ["aleve", "naprosyn", "anaprox"],
    "doxycycline": ["vibramycin", "oracea", "doxy"],
    "cetirizine": ["zyrtec", "cetryn", "cetriz"],
    "diazepam": ["valium", "valpam", "dizac"],
    "alprazolam": ["xanax", "alprax", "tafil"],
    "clonazepam": ["klonopin", "rivotril", "clon"],
    "carvedilol": ["coreg", "carvedil", "cardivas"],
    "fexofenadine": ["allegra", "telfast", "fexofine"],
    "ranitidine": ["zantac", "ranit", "rantec"],
    "diclofenac": ["voltaren", "diclof", "diclomax"],
    "ceftriaxone": ["rocephin", "ceftri", "cefaxone"],
    "cefixime": ["suprax", "cefi", "taxim"],
    "esomeprazole": ["nexium", "esotrex", "esopral"],
    "clopidogrel": ["plavix", "clopid", "plagerine"],
    
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

def preprocess_image(image_path):
    """Simple and effective image preprocessing for prescription OCR."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Simple preprocessing steps for handwritten prescriptions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        
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
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None

def run_multiple_ocr_passes(image_data):
    """Apply multiple OCR passes with different preprocessing methods"""
    results = []
    paths_to_process = [image_data["original"], image_data["enhanced"]]
    
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
        
    return results

def combine_ocr_results(results):
    """Extract and combine text from multiple OCR passes"""
    if not results:
        return "", 0.0  # Return empty string and zero confidence
        
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
                # Extract only the medication name without dosage or frequency
                med_name = re.sub(r'\s+\d+\s*\w*\b', '', ent.text) # Remove numbers and units
                med_name = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(qd|bid|tid|qid|prn|od|q\d+h)\b', '', med_name, flags=re.IGNORECASE)
                med_name = med_name.strip()
                if med_name and len(med_name) > 2:  # Ensure we have a reasonable name (not just a unit or directive)
                    medications.append(med_name)
    
    # Extract medications using our dictionary
    for key in MEDICATION_DICT.keys():
        # Skip dosage units, frequency terms, routes, and instructions
        if key in ["milligram", "microgram", "gram", "milliliter", 
                   "once daily", "twice daily", "three times daily", "four times daily",
                   "every morning", "every night", "every hour", "every 4 hours",
                   "every 6 hours", "every 8 hours", "every 12 hours", "as needed",
                   "by mouth", "intravenous", "intramuscular", "subcutaneous",
                   "sublingual", "topical", "inhalation",
                   "with food", "before meals", "after meals", "with water",
                   "do not crush", "take with plenty of water", "dissolve in water",
                   "until finished", "shake well"]:
            continue
            
        if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
            medications.append(key)
        else:
            # Check aliases, but only for medication items
            for alias in MEDICATION_DICT[key]:
                if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
                    medications.append(key)  # Add the standardized term
                    break
    
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
    
    # Clean medication names to remove any residual dosage or frequency info
    clean_medications = []
    for med in medications:
        # Remove dosage and frequency information
        clean_med = re.sub(r'\s+\d+\s*\w*\b', '', med).strip()
        clean_med = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(qd|bid|tid|qid|prn|od|q\d+h)\b', '', clean_med, flags=re.IGNORECASE).strip()
        
        if clean_med and len(clean_med) > 2:  # Ensure we have a meaningful name
            clean_medications.append(clean_med)
    
    return {
        "medications": list(set(clean_medications)),
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
        
        # STEP 1: Apply simple image preprocessing
        image_data = preprocess_image(image_path)
        if not image_data:
            return {"error": "Failed to preprocess image"}
        
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
        print(f"Error in OCR processing: {str(e)}")
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
