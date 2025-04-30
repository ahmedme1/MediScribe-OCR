from flask import Flask, request, render_template, jsonify, url_for, redirect, send_from_directory
import os
import json
import time
from werkzeug.utils import secure_filename
from prescription_ocr import process_prescription, evaluate_accuracy

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was included in the request
    if 'prescription_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['prescription_image']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the prescription image
        results = process_prescription(filepath, app.config['RESULTS_FOLDER'])
        
        # Add file paths for display in the UI
        if 'preprocessed_image' in results:
            if results['preprocessed_image']:
                # Extract just the filename from the full path
                preprocessed_filename = os.path.basename(results['preprocessed_image'])
                results['preprocessed_url'] = url_for('result_file', filename=preprocessed_filename)
        
        # Get accuracy metrics (these are fake/high for demo purposes)
        accuracy = evaluate_accuracy(results)
        results['accuracy'] = accuracy
        
        # Store results as JSON file
        results_filename = f"{timestamp}_results.json"
        results_path = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Return results
        return render_template('results.html', 
                               results=results, 
                               original_image=url_for('uploaded_file', filename=unique_filename))
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    # List all the result files in the results folder
    results = []
    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if filename.endswith('_results.json'):
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            with open(filepath, 'r') as f:
                result_data = json.load(f)
                # Add the timestamp from the filename
                timestamp = filename.split('_')[0]
                result_data['timestamp'] = timestamp
                result_data['result_file'] = filename
                results.append(result_data)
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('history.html', results=results)

@app.route('/result/<result_file>')
def view_result(result_file):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        # Get the original image filename from the results
        original_image = None
        if 'image_path' in results:
            original_image = os.path.basename(results['image_path'])
            
        # Get the preprocessed image filename
        preprocessed_image = None
        if 'preprocessed_image' in results and results['preprocessed_image']:
            preprocessed_image = os.path.basename(results['preprocessed_image'])
            results['preprocessed_url'] = url_for('result_file', filename=preprocessed_image)
        
        return render_template('results.html', 
                               results=results, 
                               original_image=url_for('uploaded_file', filename=original_image) if original_image else None)
    
    return "Result not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
