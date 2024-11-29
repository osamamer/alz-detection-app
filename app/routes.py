# app/routes.py
from app import app
from flask import render_template, jsonify

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify({"message": "Hello from Flask!"})

@app.route('/api/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Get the uploaded image
    file = request.files['file']
    image_bytes = file.read()

    # Run the image through the YOLO model
    predictions = predict(image_bytes)

    # Return the predictions as JSON
    return jsonify(predictions)