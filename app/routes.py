from flask import Blueprint, request, jsonify
from .utils import save_uploaded_file
from .model_handlers import predict_2d_image, predict_3d_image, predict_audio

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
