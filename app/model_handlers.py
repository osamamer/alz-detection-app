import logging
from .model_builder import create_model
import os
import logging
import tempfile
import numpy as np
import nibabel as nib
import ants
import antspynet
import tensorflow as tf
import shutil
import dicom2nifti
from pathlib import Path
import pydicom
from collections import defaultdict
from dicom2nifti.exceptions import ConversionValidationError, ConversionError
import os
import pandas as pd
import librosa
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from tensorflow.keras.models import load_model
import logging
from pydub import AudioSegment
from scipy.stats import pearsonr
import random
import matplotlib.pyplot as plt
import noisereduce as nr
import soundfile as sf  # For writing audio files
from scipy.signal import butter, sosfilt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import joblib


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
        Convert DICOM files to NIfTI format.

        Args:
            dicom_path: Path to DICOM file or directory containing DICOM files

        Returns:
            Path to the converted NIfTI file
        """
        try:
            logger.debug(f"Starting DICOM to NIfTI conversion from: {dicom_path}")

            # Create temporary directories for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                nifti_output = os.path.join(temp_dir, "converted.nii.gz")

                # Handle both single file and directory cases
                if os.path.isfile(dicom_path):
                    dicom_dir = os.path.join(temp_dir, "dicom")
                    os.makedirs(dicom_dir)
                    shutil.copy2(dicom_path, dicom_dir)
                    input_path = dicom_dir
                else:
                    input_path = dicom_path

                try:
                    # Perform the conversion
                    dicom2nifti.convert_directory(input_path, temp_dir, compression=True)
                except ConversionValidationError as e:
                    logger.error(f"DICOM conversion validation error: {str(e)}")
                    raise ConversionError("The uploaded DICOM files appear to be incomplete or inconsistent. Please ensure you're uploading all slices from the same series.")

                # Find the converted file
                nifti_files = list(Path(temp_dir).glob("*.nii.gz"))
                if not nifti_files:
                    raise ConversionError("Unable to convert DICOM files. Please ensure you're uploading a complete set of DICOM files from the same MRI series.")

                # Create a new temporary file for the final result
                final_output = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                shutil.copy2(str(nifti_files[0]), final_output.name)

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

#############################################################################
# AUDIO SECTION
def butter_bandpass(lowcut, highcut, sr, order=4):
    """
    Creates a Butterworth bandpass filter using second-order sections (SOS).
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(audio, sr, lowcut=300.0, highcut=4000.0, order=6):
    """
    Applies a Butterworth bandpass filter to the audio signal.
    """
    sos = butter_bandpass(lowcut, highcut, sr, order=order)
    filtered_audio = sosfilt(sos, audio)
    return filtered_audio

def reduce_noise_in_audio(audio, sr, noise_duration=0.5 ,prop_decrease = 0.9 ):
    noise_profile = audio[: int(sr * noise_duration)]  # first 0.5s as noise sample
    reduced_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile,prop_decrease = 0.9)
    return reduced_audio


def normalize_audio(audio):
    """
    Peak normalization to ensure max amplitude = 1.0
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

    return audio


def preprocess_audio(file_path, sr = 16000 , lowcut = 300 , highcut = 4000 , order = 6 , noise_duration = 0.3 , prop_decrease = 0.9):

    audio, sr = librosa.load(file_path, sr=sr)
    audio = bandpass_filter(audio, sr, lowcut=lowcut, highcut=highcut, order=order)
    audio = reduce_noise_in_audio(audio, sr, noise_duration=noise_duration , prop_decrease = prop_decrease)
    audio = normalize_audio(audio)

    return audio, sr

def extract_audio_features(audio, sr=16000):
    """
    Extract multiple audio features from an audio signal (numpy array).

    Parameters:
    - audio (numpy.ndarray): The audio signal.
    - sr (int): Sample rate of the audio signal.

    Returns:
    - combined_features (numpy.ndarray): Combined features extracted from the audio.
    """
    try:
        features = {}

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)

        # Mel-Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        features['mel_mean'] = np.mean(mel_spectrogram, axis=1)
        features['mel_std'] = np.std(mel_spectrogram, axis=1)

        # Spectral Features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=audio))
        features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(audio)), sr=sr))

        # Temporal Features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_strength_mean'] = np.mean(onset_env)

        # Energy-Based Features
        rmse = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rmse)
        features['rms_std'] = np.std(rmse)

        energy_entropy = -np.sum(rmse * np.log(rmse + 1e-6))
        features['energy_entropy'] = energy_entropy

        # Spectral Flux
        spectral_flux = np.mean(librosa.onset.onset_strength(y=audio, sr=sr, lag=1))
        features['spectral_flux'] = spectral_flux

        # Spectral Variance
        features['spectral_variance'] = np.var(librosa.feature.spectral_centroid(y=audio, sr=sr))

        # Zero Crossing Rate
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # RMS Energy
        features['rms'] = np.mean(librosa.feature.rms(y=audio))

        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)

        # Linguistic Features: Pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        features['mean_pitch'] = pitch_values.mean() if len(pitch_values) > 0 else 0
        features['stdev_pitch'] = pitch_values.std() if len(pitch_values) > 0 else 0

        # Speaking rate
        non_silent_intervals = librosa.effects.split(audio, top_db=10)
        duration = sum((end - start) for start, end in non_silent_intervals) / sr
        speaking_rate = len(non_silent_intervals) / duration
        features['speaking_rate'] = speaking_rate

        # Pause duration
        total_pause_duration = (len(audio) - sum((end - start) for start, end in non_silent_intervals)) / sr
        features['pause_duration'] = total_pause_duration

        # Combine all features into a single array
        combined_features = np.concatenate([
            features[key] if isinstance(features[key], np.ndarray) else [features[key]]
            for key in features
        ])

        return combined_features

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def combine_features(metadata, data_path):
    """
    Combine audio features with metadata (age, gender, educ).
    This version tries .mp3 first, then .wav if .mp3 not found.
    """
    features = []
    labels = []

    for index, row in metadata.iterrows():
        file_name = row['adressfname']  # e.g., "adrso002"

        # 1) Try .mp3 first
        file_path_mp3 = os.path.join(data_path, file_name + ".mp3")
        file_path_wav = os.path.join(data_path, file_name + ".wav")

        # 2) Determine which file actually exists
        if os.path.isfile(file_path_mp3):
            file_path = file_path_mp3
        elif os.path.isfile(file_path_wav):
            file_path = file_path_wav
        else:
            print(f"File not found (neither .mp3 nor .wav) for: {file_name}")
            continue

        # 3) Extract features
        audio_features = extract_audio_features(file_path)
        if audio_features is not None:
            # Normalize and encode metadata features
            age = row['age'] / 100.0    # e.g., scale age by /100
            gender = 1 if row['gender'].lower() == 'male' else 0

            combined_features = np.concatenate((audio_features, [age, gender]))
            features.append(combined_features)
            labels.append(row['dx'])

    return np.array(features), np.array(labels)

def predict_with_fitted_scaling(
        file_path, age, gender, model, significant_features, scaler,
        sr=16000, lowcut=300, highcut=4000, order=6, noise_duration=0.5, prop_decrease=0.9
):
    """
    Predicts the class of a single audio file by fitting the imputer and scaler
    inside the function, ensuring that the features align with the model input.

    Parameters:
    - file_path (str): Path to the audio file.
    - age (float): Age of the subject.
    - gender (str): Gender of the subject ("male" or "female").
    - model (keras.Model): Loaded trained model.
    - significant_features (list): List of 96 significant features.
    - sr, lowcut, highcut, order, noise_duration, prop_decrease: Preprocessing parameters.

    Returns:
    - prediction (int): Predicted class label.
    - confidence (float): Confidence of the prediction (max probability).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Step 1: Preprocess the audio
    try:
        processed_audio, _ = preprocess_audio(
            file_path, sr=sr, lowcut=lowcut, highcut=highcut,
            order=order, noise_duration=noise_duration, prop_decrease=prop_decrease
        )
    except Exception as e:
        raise ValueError(f"Error in audio preprocessing: {e}")

    # Step 2: Extract features from processed audio
    extracted_features = extract_audio_features(audio=processed_audio, sr=sr)
    if extracted_features is None:
        raise ValueError("Feature extraction failed for the given audio file.")

    # Step 3: Combine features with metadata (age, gender)
    age_normalized = age / 100.0  # Normalize age
    gender_encoded = 1 if gender.lower() == 'male' else 0  # Encode gender
    combined_features = np.append(extracted_features, [age_normalized, gender_encoded])

    # Step 4: Create a DataFrame for filtering significant features
    feature_names = [f"feature_{i}" for i in range(len(combined_features))]
    feature_df = pd.DataFrame([combined_features], columns=feature_names)

    # Step 5: Filter only the significant features
    try:
        filtered_features = feature_df[significant_features].to_numpy()
    except KeyError as e:
        raise ValueError(f"Feature mismatch: {e}")


    # Step 7: Fit and apply scaling
    scaled_features = scaler.transform(filtered_features)

    # Step 8: Reshape for the model
    reshaped_features = np.expand_dims(scaled_features, axis=-1)  # Shape: (1, num_features, 1)

    # Step 9: Predict using the model
    prediction_probs = model.predict(reshaped_features)
    prediction = np.argmax(prediction_probs, axis=1)[0]  # Predicted class label
    confidence = np.max(prediction_probs)  # Confidence (highest probability)

    # Map class index to label
    class_mapping = {0: "Control", 1: "ProbableAD"}
    predicted_class_label = class_mapping[prediction]
    print(predicted_class_label)
    print(confidence)
    return predicted_class_label, confidence


def predict_audio(file_path, age, gender):
    """
    Predicts the class of an audio file.

    Returns:
    - tuple: (predicted_class_label, confidence)
    """
    try:
        audio_model = tf.keras.models.load_model('app/models/best_model_english.keras')
        scaler = joblib.load('app/models/scaler.pkl')
        significant_features = joblib.load('app/models/significant_features.pkl')

        # Load and preprocess audio
        try:
            preprocessed_audio, sr = preprocess_audio(file_path)
        except Exception as e:
            raise Exception(f"Error preprocessing audio: {str(e)}")

        # Make prediction
        predicted_class_label, confidence = predict_with_fitted_scaling(
            file_path, age, gender, audio_model, significant_features, scaler
        )

        return predicted_class_label, confidence

    except Exception as e:
        raise Exception(f"Error in prediction pipeline: {str(e)}")
