from flask import Blueprint, request, jsonify
from .utils import save_uploaded_file
from .model_handlers import predict_2d_image, predict_3d_image, predict_audio
import base64
import numpy as np
from PIL import Image
from io import BytesIO

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/predict-2d-image', methods=['POST'])
def predict_2d():
    image_file = request.files['image']
    file_path = save_uploaded_file(image_file)
    result = predict_2d_image(file_path)
    return jsonify({'result': result})

@main_routes.route('/predict-3d-image', methods=['POST'])
def predict_3d():
    image_file = request.files['image']
    file_path = save_uploaded_file(image_file)
    result = predict_3d_image(file_path)
    return jsonify({'result': result})

@main_routes.route('/predict-audio', methods=['POST'])
def predict_audio_route():
    audio_file = request.files['audio']
    file_path = save_uploaded_file(audio_file)
    result = predict_audio(file_path)
    return jsonify({'result': result})

@main_routes.route('/analyze-picture', methods=['POST'])
def analyze_picture():
    data = request.json
    description = data.get('description', '')

    if not description:
        return jsonify({'result': 'No description provided. Please try again.'}), 400

    # Simple mock NLP analysis
    keywords = ["plate", "sink", "overflow", "woman"]
    keyword_matches = [word for word in keywords if word in description.lower()]
    if keyword_matches:
        result = f"Good job! You noticed these key elements: {', '.join(keyword_matches)}."
    else:
        result = "Try to provide more details in your description."

    return jsonify({'result': result})

@main_routes.route('/analyze-clock', methods=['POST'])
def analyze_clock():
    data = request.json
    drawing_data = data.get('drawingData')

    if not drawing_data:
        return jsonify({'result': 'No drawing data provided. Please try again.'}), 400

    # Decode drawing data
    drawing_image = decode_drawing(drawing_data)

    # Mock image analysis
    result = "The clock drawing looks good! Make sure numbers and hands are clear."

    # Save the drawing for further analysis (optional)
    drawing_image.save('uploads/clock_drawing.png')

    return jsonify({'result': result})

def decode_drawing(drawing_data):
    """Decode Base64 drawing data into a PIL image."""
    decoded_data = base64.b64decode(drawing_data)
    buffer = BytesIO(decoded_data)
    image = Image.open(buffer)
    return image

@main_routes.route('/evaluate-memory', methods=['POST'])
def evaluate_memory():
    data = request.json
    user_input = data.get('user_input', '')  # Words entered by the user
    word_list = ['apple', 'house', 'river', 'chair', 'dog']  # Original word bank

    if not user_input:
        return jsonify({'result': 'No words provided. Please try again.'}), 400

    # Process user input
    user_words = {word.strip().lower() for word in user_input.split(',')}
    correct_words = set(word_list)
    correct_count = len(user_words & correct_words)

    result = {
        'score': correct_count,
        'total': len(word_list),
        'feedback': f"You remembered {correct_count} out of {len(word_list)} words."
    }

    return jsonify(result)
