from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_api():
    file = request.files['image']
    text = ocr_easyocr(file)
    corrected_text = correct_text(text)
    return jsonify({"extracted_text": corrected_text})

if __name__ == '__main__':
    app.run(debug=True)
