from flask import Blueprint, request, jsonify
from .utils import save_uploaded_file
from .model_handlers import predict_2d_image, predict_3d_image, predict_audio
import base64
import numpy as np
# import tensorflow as tf
from PIL import Image
from io import BytesIO
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
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

# Load Hugging Face pipeline
analyzer = pipeline("text2text-generation", model="google/flan-t5-small", framework="pt")

@main_routes.route('/analyze-picture', methods=['POST'])
def analyze_picture():
    data = request.json
    description = data.get('description', '')

    if not description:
        return jsonify({'result': 'No description provided. Please try again.'}), 400

    prompt = f"""
    You are a language analysis assistant. Analyze the following description of a picture:
    Description: "{description}"
    
    Key elements to look for:
    - Coherence: Does the description flow logically? Are the sentences well-structured and connected?
    - Detail: How detailed is the description? Does it include specific observations?
    - Vocabulary: Are relevant and appropriate terms used?

    Provide the analysis in this structured format:
    - Coherence (0-3): [Score and explanation]
    - Detail (0-3): [Score and explanation]
    - Vocabulary (0-3): [Score and explanation]
    - Key elements mentioned: [List of elements]
    - Overall feedback: [Summary of strengths and areas for improvement]
    """
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    input_text = f"""
    Please return a text output. Analyze the following description of a picture:
    Description: "{description}"
    
    Key elements to look for:
    - Coherence: Does the description flow logically? Are the sentences well-structured and connected?
    - Detail: How detailed is the description? Does it include specific observations?
    - Vocabulary: Are relevant and appropriate terms used?

    Provide the analysis in this structured format:
    - Coherence (0-3): [Score and explanation]
    - Detail (0-3): [Score and explanation]
    - Vocabulary (0-3): [Score and explanation]
    - Key elements mentioned: [List of elements]
    - Overall feedback: [Summary of strengths and areas for improvement]
    """
    # input_text = "What is a strawberry?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    # response = analyzer(prompt, max_length=200)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])

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
