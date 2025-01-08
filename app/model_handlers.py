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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ModelHandler class for processing and predicting
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
