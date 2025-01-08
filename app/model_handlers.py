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
    Input, Dropout, Add, multiply
)
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
def create_se_block(inputs, filters, reduction_ratio=4):
    """Squeeze-and-Excitation block adapted for 3D"""
    se = GlobalAveragePooling3D()(inputs)
    se = Dense(filters // reduction_ratio, activation='swish')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = tf.reshape(se, [-1, 1, 1, 1, filters])
    return multiply([inputs, se])

def create_mbconv_block(inputs, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                        expansion_factor=6, se_ratio=4, dropout_rate=0.3):
    """Enhanced MBConv block with Squeeze-and-Excitation"""
    input_filters = inputs.shape[-1]
    expanded_filters = input_filters * expansion_factor

    # Expansion phase
    if expansion_factor != 1:
        x = Conv3D(expanded_filters, (1, 1, 1), padding="same", use_bias=False)(inputs)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('swish')(x)
    else:
        x = inputs

    # Depthwise convolution
    x = Conv3D(expanded_filters, kernel_size, strides=strides,
               padding="same", groups=expanded_filters, use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('swish')(x)

    # Squeeze-and-Excitation
    x = create_se_block(x, expanded_filters, se_ratio)

    # Projection phase
    x = Conv3D(filters, (1, 1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)

    # Skip connection
    if strides == (1, 1, 1) and input_filters == filters:
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Add()([x, inputs])

    return x



class ModelHandler:
    def __init__(self, model_path, mni_template_path, target_shape=(128, 128, 128)):
        logger.debug(f"Initializing ModelHandler with model_path={model_path}, template_path={mni_template_path}")
        try:
            # Define custom objects
            custom_objects = {
                'create_se_block': create_se_block,
                'create_mbconv_block': create_mbconv_block,
                'swish': tf.keras.activations.swish,
                'AdamW': tf.keras.optimizers.AdamW
            }

            # Load model with custom objects
            self.model = keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )
            logger.debug("Model loaded successfully")

            # Verify template path exists
            if not os.path.exists(mni_template_path):
                raise FileNotFoundError(f"MNI template not found at: {mni_template_path}")

            self.mni_template_path = mni_template_path
            self.target_shape = target_shape

        except Exception as e:
            logger.error(f"Error in ModelHandler initialization: {str(e)}")
            raise

    def process_single_mri(self, input_path, output_path):
        """Process a single MRI image using ANTS"""
        try:
            logger.debug(f"Starting MRI processing for file: {input_path}")

            # Verify input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

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

            # Verify output was created
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Failed to create output file: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error in process_single_mri: {str(e)}")
            raise

    def predict_3d_image(self, file_path):
        """Process and predict on a 3D MRI image"""
        try:
            logger.debug(f"Starting prediction pipeline for file: {file_path}")

            # Create a temporary directory for processed image
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.debug(f"Created temporary directory: {temp_dir}")
                processed_path = os.path.join(temp_dir, 'processed_mri.nii.gz')

                # Process the MRI using ANTS
                logger.debug("Starting MRI processing...")
                processed_path = self.process_single_mri(file_path, processed_path)
                logger.debug("MRI processing completed")

                # Prepare for prediction
                logger.debug("Preparing image for prediction...")
                prepared_image = self.prepare_for_prediction(processed_path)
                logger.debug(f"Prepared image shape: {prepared_image.shape}")

                # Make prediction
                logger.debug("Making prediction...")
                prediction = self.model.predict(prepared_image)
                logger.debug(f"Raw prediction: {prediction}")

                # Convert prediction to class label
                class_names = ['AD', 'MCI', 'CN']
                predicted_class = class_names[np.argmax(prediction[0])]
                probabilities = {class_names[i]: float(pred) for i, pred in enumerate(prediction[0])}

                logger.debug(f"Predicted class: {predicted_class}")
                logger.debug(f"Probabilities: {probabilities}")

                return {
                    'predicted_class': predicted_class,
                    'probabilities': probabilities
                }

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            raise

def predict_3d_image(file_path):
    logger.debug(f"Called predict_3d_image with file_path: {file_path}")
    try:
        model_handler = ModelHandler(
            model_path='app/models/Best-3D-Model.h5',
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


def predict_3d_image(file_path):
    model_handler = ModelHandler(
        model_path='app/models/Best-3D-Model.h5',
        mni_template_path='app/models/mni_icbm152_t1_tal_nlin_sym_09c.nii',  # Adjust path as needed
        target_shape=(128, 128, 128)
    )
    prediction = model_handler.predict_3d_image(file_path)
    return prediction

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
