from dicom2nifti.exceptions import ConversionValidationError
from flask import Blueprint, request, jsonify, current_app
from .utils import save_uploaded_file
from .model_handlers import predict_3d_image, predict_audio
import base64
from PIL import Image
from io import BytesIO
import os
from werkzeug.utils import secure_filename
import tempfile
import logging
class ConversionError(Exception):
    pass
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  # This will create a logger specific to this file

main_routes = Blueprint('main_routes', __name__)


@main_routes.route('/predict-3d-image', methods=['POST'])
def predict_3d():
    logger.debug("Received predict-3d-image request")
    try:
        # Check for NIfTI file
        if 'image' in request.files:
            image_file = request.files['image']
            logger.debug(f"Received NIfTI file: {image_file.filename}")

            # Check if file was selected
            if image_file.filename == '':
                logger.error("No file selected")
                return jsonify({'error': 'No file selected'}), 400

            # Verify file type
            if not image_file.filename.endswith(('.nii', '.nii.gz')):
                logger.error(f"Invalid file type: {image_file.filename}")
                return jsonify({'error': 'Invalid file type. Please upload a .nii or .nii.gz file'}), 400

            # Create temporary directory and save file
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.debug(f"Created temporary directory: {temp_dir}")
                filename = secure_filename(image_file.filename)
                file_path = os.path.join(temp_dir, filename)

                # Save uploaded file
                image_file.save(file_path)
                logger.debug(f"Saved uploaded file to: {file_path}")

                # Process the image and get prediction
                result = predict_3d_image(file_path)
                logger.debug(f"Prediction complete: {result}")

                return jsonify({'result': result})

        # Check for DICOM files
        elif 'dicom_files[]' in request.files:
            files = request.files.getlist('dicom_files[]')
            logger.debug(f"Received {len(files)} DICOM files")

            if not files:
                logger.error("No DICOM files uploaded")
                return jsonify({'error': 'No DICOM files uploaded'}), 400

            # Create temporary directory and save files
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.debug(f"Created temporary directory: {temp_dir}")

                # Save all DICOM files to temporary directory
                dicom_dir = os.path.join(temp_dir, 'dicom_files')
                os.makedirs(dicom_dir)

                for dicom_file in files:
                    if dicom_file.filename == '':
                        continue
                    if not dicom_file.filename.lower().endswith(('.dcm', '.dicom')):
                        logger.error(f"Invalid file type: {dicom_file.filename}")
                        return jsonify({'error': f'Invalid file type for {dicom_file.filename}. Please upload only DICOM files'}), 400

                    filename = secure_filename(dicom_file.filename)
                    file_path = os.path.join(dicom_dir, filename)
                    dicom_file.save(file_path)
                    logger.debug(f"Saved DICOM file to: {file_path}")

                try:
                    # Process the DICOM directory and get prediction
                    result = predict_3d_image(dicom_dir)
                    logger.debug(f"Prediction complete: {result}")
                    return jsonify({'result': result})

                except ConversionValidationError as e:
                    error_message = "The uploaded DICOM files appear to be incomplete or inconsistent. Please ensure you're uploading all slices from the same series in the correct sequence. A complete brain MRI series typically contains consecutive slices with consistent spacing."
                    logger.error(f"DICOM conversion error: {error_message}")
                    return jsonify({'error': error_message}), 400
                except Exception as e:
                    if "No NIfTI file was created during conversion" in str(e):
                        error_message = "Unable to convert DICOM files. Please ensure you're uploading a complete set of DICOM files from the same MRI series. All slices should be from the same scan sequence."
                        logger.error(f"DICOM conversion error: {error_message}")
                        return jsonify({'error': error_message}), 400
                    raise

        else:
            logger.error("No files found in request")
            return jsonify({'error': 'No files uploaded'}), 400

    except Exception as e:
        logger.error(f"Server error in predict_3d route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main_routes.route('/predict-audio', methods=['POST'])
def predict_audio_route():
    try:
        # Validate that all required fields are present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        if 'age' not in request.form:
            return jsonify({'error': 'Age is required'}), 400

        if 'gender' not in request.form:
            return jsonify({'error': 'Gender is required'}), 400

        # Get the files and form data
        audio_file = request.files['audio']
        try:
            age = float(request.form['age'])  # Convert to float since model expects float
            if age < 18 or age > 120:
                return jsonify({'error': 'Age must be between 18 and 120'}), 400
        except ValueError:
            return jsonify({'error': 'Age must be a valid number'}), 400

        gender = request.form['gender'].lower()
        valid_genders = ['male', 'female', 'other']
        if gender not in valid_genders:
            return jsonify({'error': 'Gender must be one of: male, female, other'}), 400

        # Save file with secure filename
        filename = secure_filename(audio_file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400

        # Ensure file has correct extension
        if not filename.lower().endswith(('.mp3', '.wav')):
            return jsonify({'error': 'Only MP3 and WAV files are allowed'}), 400

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(current_app.root_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Save file temporarily
        file_path = os.path.join(temp_dir, filename)
        audio_file.save(file_path)

        try:
            # Make prediction
            predicted_class, confidence = predict_audio(file_path, age, gender)

            # Clean up temp file
            os.remove(file_path)

            return jsonify({
                'result': {
                    'prediction': predicted_class,
                    'confidence': float(confidence),  # Convert numpy float to Python float
                    'age': age,
                    'gender': gender
                }
            })

        except Exception as pred_error:
            # Clean up temp file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            current_app.logger.error(f"Prediction error: {str(pred_error)}")
            return jsonify({'error': f'Error processing audio: {str(pred_error)}'}), 500

    except Exception as e:
        current_app.logger.error(f"Route error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
