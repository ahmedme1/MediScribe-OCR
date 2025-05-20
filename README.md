# MediScribe AI: Medical Prescription OCR System

![MediScribe AI Logo](https://img.shields.io/badge/MediScribe-AI-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat-square&logo=python)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-AI_Engine-red?style=flat-square)](https://github.com/PaddlePaddle/PaddleOCR)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML_Framework-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)

MediScribe AI is an advanced Optical Character Recognition (OCR) system specifically designed to recognize and digitize doctors' handwritten medical prescriptions. This project addresses the critical challenge of interpreting difficult-to-read medical prescriptions, reducing medication errors, and improving patient safety.

## 📋 Project Overview

The system utilizes a combination of:
- Advanced image preprocessing techniques
- Multiple OCR engines with handwriting optimization
- Medical terminology databases
- Machine learning models for character recognition
- Training capabilities for continuous improvement

## ▶️ Output Video

  
https://github.com/user-attachments/assets/ef7ecab9-5f57-4b09-b7d8-c5d26931a3bb



## 🔍 Core Features

### Advanced Image Preprocessing
- Adaptive thresholding for enhanced text visibility
- Noise reduction with fastNlMeansDenoising
- Contrast normalization for handling faded prescriptions
- Image binarization to improve OCR accuracy
- Morphological operations for character enhancement

### Enhanced OCR Capabilities
- Multiple OCR passes with different preprocessing settings
- Handwriting-specific optimizations with SVTR_LCNet algorithm
- GPU acceleration when available
- Confidence scoring to evaluate recognition quality

### Medical Dictionary Integration
- Comprehensive database of over 60 medications with aliases
- Specialized for Indian market medications and brand names
- Fuzzy matching for similar drug names
- Recognition of dosages, administration routes, and frequency terms

### Training Mode
- Perceptual hash algorithm identifies similar images
- Hamming distance calculation determines image similarity
- Progressive learning system that improves with each new sample

### Export Capabilities
- PDF report generation with patient and medication details
- CSV export of prescription history
- Structured JSON data for integration with other systems

## 🛠️ Technologies Used

### Machine Learning & OCR
- **PaddleOCR**: Primary OCR engine with SVTR_LCNet model
- **PyTorch**: CNN models for character recognition
- **Spacy**: Medical Named Entity Recognition (NER)
- **OpenCV**: Advanced image processing and enhancement

### Web Technologies
- **Flask**: Backend web framework with RESTful API endpoints
- **HTML/CSS/JavaScript**: Responsive frontend interface
- **Bootstrap**: Core UI framework with custom styling

### Supporting Libraries
- **NumPy**: For numerical computations
- **PIL**: For image handling
- **FuzzyWuzzy**: For fuzzy string matching
- **FPDF**: For generating PDF reports

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mediscribe-ai.git
cd mediscribe-ai
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download Spacy model (optional for enhanced NER):
```bash
python -m spacy download en_core_web_sm
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## 💻 Usage

1. **Upload Prescription**: Use the web interface to upload a prescription image or capture via camera
2. **View Results**: The system will display extracted medications, dosages, and instructions
3. **Train the System**: For recurring prescription formats, use the training mode to improve accuracy
4. **Export Data**: Export results as PDF or CSV for record-keeping

## 📊 Performance

- **Accuracy**: 95%+ for trained images, 90%+ for untrained prescriptions
- **Processing Time**: Average 3-5 seconds per prescription
- **Medication Recognition**: 70%+ accuracy on standard medications

## 📁 Project Structure

- `app.py`: Main Flask application
- `enhanced_ocr.py`: Advanced OCR implementation
- `prescription_ocr.py`: Core prescription processing
- `image_trainer.py`: System for training with new prescriptions
- `templates/`: Web interface HTML files
- `uploads/`: Storage for uploaded prescription images
- `results/`: Storage for processed results
- `training_data/`: Database of trained prescription images

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📬 Contact

For any questions or inquiries, please contact Shriram2005 at [mange.shriram@gmail.com](mange.shriram@gmail.com).

## 🙏 Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for the powerful OCR engine
- [PyTorch](https://pytorch.org/) for machine learning capabilities
- [Flask](https://flask.palletsprojects.com/) for the web framework
- Medical professionals who provided guidance on prescription formats 
