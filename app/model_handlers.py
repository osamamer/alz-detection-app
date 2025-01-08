import keras
from tensorflow.keras.models import load_model

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

import ants
import antspynet
import numpy as np
# from tensorflow import keras
import os
import tempfile
import nibabel as nib
import tensorflow as tf
import keras
from keras import layers
from keras.layers import (
    Conv3D, BatchNormalization, Activation, Dense, GlobalAveragePooling3D,
    Input, Dropout, Add, multiply, Lambda
)
import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import nibabel as nib
import antspynet
import numpy as np
import logging
import os
import logging
import tempfile
import numpy as np
import nibabel as nib
import ants
import antspynet
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from .model_builder import create_model
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ModelHandler class for processing and predicting
import os
import logging
import tempfile
import numpy as np
import nibabel as nib
import ants
import antspynet
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import shutil
import dicom2nifti
from pathlib import Path
import pydicom
from collections import defaultdict
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, weights_path, mni_template_path, target_shape=(128, 128, 128)):
        logger.debug(f"Initializing ModelHandler with weights_path={weights_path}, template_path={mni_template_path}")
        try:
            # Rebuild the model architecture and load weights
            self.model = self.build_model()
            self.model.load_weights(weights_path)
            logger.debug("Model architecture rebuilt and weights loaded successfully")

            if not os.path.exists(mni_template_path):
                raise FileNotFoundError(f"MNI template not found at: {mni_template_path}")

            self.mni_template_path = mni_template_path
            self.target_shape = target_shape

        except Exception as e:
            logger.error(f"Error in ModelHandler initialization: {str(e)}")
            raise

    def build_model(self):
        return create_model()

    def organize_dicom_series(self, dicom_path):
        """
        Organize DICOM files into series, handling multiple files properly.

        Args:
            dicom_path: Path to directory containing DICOM files

        Returns:
            Dictionary mapping series UIDs to lists of sorted DICOM files
        """
        logger.debug(f"Organizing DICOM series from: {dicom_path}")
        series_dict = defaultdict(list)

        # Handle both single file and directory cases
        if os.path.isfile(dicom_path):
            dicom_files = [dicom_path]
        else:
            dicom_files = [os.path.join(dicom_path, f) for f in os.listdir(dicom_path)
                           if f.lower().endswith(('.dcm', '.dicom'))]

        for file_path in dicom_files:
            try:
                dcm = pydicom.dcmread(file_path)
                # Group by SeriesInstanceUID
                series_dict[dcm.SeriesInstanceUID].append((file_path, dcm))
            except Exception as e:
                logger.warning(f"Skipping invalid DICOM file {file_path}: {str(e)}")

        # Sort each series by slice location or instance number
        for series_uid in series_dict:
            series_dict[series_uid].sort(
                key=lambda x: (getattr(x[1], 'SliceLocation', 0),
                               getattr(x[1], 'InstanceNumber', 0))
            )

        return series_dict

    def convert_dicom_to_nifti(self, dicom_path):
        """
        Convert organized DICOM series to NIfTI format with preprocessing.

        Args:
            dicom_path: Path to DICOM file or directory containing DICOM files

        Returns:
            Path to the converted and preprocessed NIfTI file
        """
        try:
            logger.debug(f"Starting DICOM to NIfTI conversion from: {dicom_path}")

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Organize DICOM files into series
                series_dict = self.organize_dicom_series(dicom_path)

                if not series_dict:
                    raise ValueError("No valid DICOM series found")

                # Process each series (usually there should be one T1 series)
                for series_uid, dicom_files in series_dict.items():
                    # Create a subdirectory for this series
                    series_dir = os.path.join(temp_dir, f"series_{series_uid}")
                    os.makedirs(series_dir)

                    # Copy DICOM files to the series directory
                    for file_path, _ in dicom_files:
                        shutil.copy2(file_path, series_dir)

                    # Convert to NIfTI
                    nifti_output = os.path.join(temp_dir, f"converted_{series_uid}.nii.gz")
                    dicom2nifti.convert_directory(
                        series_dir,
                        temp_dir,
                        compression=True,
                        reorient=True  # Ensure proper orientation
                    )

                # Find the converted file(s)
                nifti_files = list(Path(temp_dir).glob("*.nii.gz"))
                if not nifti_files:
                    raise ValueError("No NIfTI file was created during conversion")

                # If multiple series were converted, use the largest one (assuming it's the full brain T1)
                nifti_file = max(nifti_files, key=lambda f: os.path.getsize(f))

                # Create a new temporary file for the final result
                final_output = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                shutil.copy2(str(nifti_file), final_output.name)

                logger.debug(f"DICOM conversion successful, saved to: {final_output.name}")
                return final_output.name

        except Exception as e:
            logger.error(f"Error in DICOM to NIfTI conversion: {str(e)}")
            raise

    def predict_from_dicom(self, dicom_path):
        """
        Predict directly from DICOM file(s), including preprocessing

        Args:
            dicom_path: Path to DICOM file or directory containing DICOM files

        Returns:
            Dictionary containing prediction results
        """
        try:
            logger.debug(f"Starting prediction from DICOM: {dicom_path}")

            # Convert DICOM to NIfTI
            nifti_path = self.convert_dicom_to_nifti(dicom_path)

            try:
                # Process the NIfTI file (registration, brain extraction, etc.)
                processed_path = self.process_single_mri(nifti_path, nifti_path)

                # Use existing prediction pipeline
                return self.predict_3d_image(processed_path)
            finally:
                # Clean up temporary files
                if os.path.exists(nifti_path):
                    os.unlink(nifti_path)

        except Exception as e:
            logger.error(f"Error in predict_from_dicom: {str(e)}")
            raise

    def normalize_volume(self, volume):
        """Normalize the volume to zero mean and unit variance"""
        mean = tf.reduce_mean(volume)
        variance = tf.reduce_mean(tf.square(volume - mean))
        std = tf.sqrt(variance)
        return tf.cond(
            tf.equal(std, 0),
            lambda: volume - mean,
            lambda: (volume - mean) / std
        )

    def resize_volume(self, volume):
        """Resize volume to target shape using tf.image operations"""
        # Ensure input is 3D
        if len(volume.shape) == 4:
            volume = tf.squeeze(volume, axis=0)

        # Resize if needed
        if volume.shape != self.target_shape:
            volume = tf.expand_dims(volume, axis=0)
            volume = tf.image.resize(volume, self.target_shape[:2])
            volume = tf.transpose(volume, [0, 3, 2, 1])
            volume = tf.image.resize(volume, self.target_shape[:2])
            volume = tf.transpose(volume, [0, 3, 2, 1])
            volume = tf.squeeze(volume, axis=0)

        return volume

    def process_single_mri(self, input_path, output_path):
        """Process a single MRI image using ANTS"""
        try:
            logger.debug(f"Starting MRI processing for file: {input_path}")

            logger.debug("Loading images...")
            moving_image = ants.image_read(input_path)
            logger.debug(f"Moving image loaded, shape: {moving_image.shape}")

            fixed_image = ants.image_read(self.mni_template_path)
            logger.debug(f"Fixed image loaded, shape: {fixed_image.shape}")

            logger.debug("Performing N4 bias field correction...")
            moving_image = ants.n4_bias_field_correction(moving_image)

            logger.debug("Performing denoising...")
            moving_image = ants.denoise_image(moving_image)

            logger.debug("Computing initial transform...")
            init_transform = ants.affine_initializer(fixed_image, moving_image)

            logger.debug("Starting registration...")
            registration = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                initial_transform=init_transform,
                type_of_transform='AffineFast',
                transform_parameters=(0.1, 3, 0),
                similarity_metric='CC',
                shrink_factors=[4, 2, 1],
                smoothing_sigmas=[2, 1, 0],
                verbose=True
            )
            logger.debug("Registration completed")

            logger.debug("Performing brain extraction...")
            brain_mask = antspynet.brain_extraction(
                registration['warpedmovout'],
                modality='t1'
            )
            final_image = registration['warpedmovout'] * brain_mask

            logger.debug(f"Saving result to {output_path}")
            ants.image_write(final_image, output_path)
            return output_path

        except Exception as e:
            logger.error(f"Error in process_single_mri: {str(e)}")
            raise

    def prepare_for_prediction(self, file_path):
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
            try:
                processed_path = self.process_single_mri(file_path, temp_file.name)
                # Load the processed image
                img = nib.load(processed_path)
                volume = img.get_fdata()
                volume = tf.convert_to_tensor(volume, dtype=tf.float32)
                volume = self.resize_volume(volume)
                volume = self.normalize_volume(volume)
                volume = tf.expand_dims(volume, axis=-1)
                volume = tf.expand_dims(volume, axis=0)  # Add batch dimension
                return volume
            finally:
                temp_file.close()
                os.unlink(temp_file.name)  # Clean up the temporary file
        except Exception as e:
            logger.error(f"Error in prepare_for_prediction: {str(e)}")
            raise


    def predict_3d_image(self, file_path):
        try:
            # Prepare image for prediction
            prepared_image = self.prepare_for_prediction(file_path)
            prediction = self.model.predict(prepared_image, verbose=0)
            class_names = ['AD', 'CN', 'MCI']
            predicted_class = class_names[np.argmax(prediction[0])]
            probabilities = {class_names[i]: float(pred) for i, pred in enumerate(prediction[0])}
            return {'predicted_class': predicted_class, 'probabilities': probabilities}
        except Exception as e:
            logger.error(f"Error in predict_3d_image: {str(e)}")
            raise



# Wrapper function
def predict_3d_image(file_path):
    logger.debug(f"Called predict_3d_image with file_path: {file_path}")
    try:
        model_handler = ModelHandler(
            weights_path='app/models/Best-3D-Model.h5',
            mni_template_path='app/models/mni_icbm152_t1_tal_nlin_sym_09c.nii',
            target_shape=(128, 128, 128)
        )

        # Check if input is DICOM
        if file_path.lower().endswith(('.dcm', '.dicom')) or (
                os.path.isdir(file_path) and any(f.lower().endswith(('.dcm', '.dicom'))
                                                 for f in os.listdir(file_path))):
            prediction = model_handler.predict_from_dicom(file_path)
        else:
            prediction = model_handler.predict_3d_image(file_path)

        logger.debug(f"Prediction successful: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error in predict_3d_image wrapper: {str(e)}")
        raise


# model_audio = load_model('app/models/dummy_model.keras')



def predict_2d_image(file_path):
    # Add preprocessing logic for 2D images here
    # Example:
    # image_data = preprocess_2d_image(file_path)
    # prediction = model_2d.predict(image_data)
    prediction = "Mock prediction for 2D image"  # Replace with real prediction
    return prediction
def predict_audio(file_path):
    # Add preprocessing logic for audio files here
    # Example:
    # audio_data = preprocess_audio(file_path)
    # prediction = model_audio.predict(audio_data)
    prediction = "Mock prediction for audio"  # Replace with real prediction
    return prediction
