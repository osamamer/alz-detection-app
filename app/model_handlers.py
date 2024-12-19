from tensorflow.keras.models import load_model
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
import tensorflow as tf

# Load the model


# Load models at the start
model_2d = load_model('/home/osama/AlzieDet/App/app/models/dummy_model.keras')
model_3d = load_model('/home/osama/AlzieDet/App/app/models/dummy_model.keras')
model_audio = load_model('/home/osama/AlzieDet/App/app/models/dummy_model.keras')

def predict_2d_image(file_path):
    # Add preprocessing logic for 2D images here
    # Example:
    # image_data = preprocess_2d_image(file_path)
    # prediction = model_2d.predict(image_data)
    prediction = "Mock prediction for 2D image"  # Replace with real prediction
    return prediction

def predict_3d_image(file_path):
    # Add preprocessing logic for 3D images here
    # Example:
    # image_data = preprocess_3d_image(file_path)
    # prediction = model_3d.predict(image_data)
    prediction = "Mock prediction for 3D image"  # Replace with real prediction
    return prediction

def predict_audio(file_path):
    # Add preprocessing logic for audio files here
    # Example:
    # audio_data = preprocess_audio(file_path)
    # prediction = model_audio.predict(audio_data)
    prediction = "Mock prediction for audio"  # Replace with real prediction
    return prediction
