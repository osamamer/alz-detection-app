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
from transformers import pipeline
import spacy
import re
from collections import Counter

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




###################################################################################################################

# Load Hugging Face pipeline
# analyzer = pipeline("text2text-generation", model="google/flan-t5-small", framework="pt")

from transformers import pipeline
import spacy
from collections import Counter

from transformers import pipeline
import spacy
from collections import Counter

class CookieTheftAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # Essential elements based on clinical criteria
        self.essential_elements = {
            'subject_items': [
                'mother', 'woman', 'lady',             # Mother figure
                'boy', 'boys', 'children', 'kid',      # Children
                'kitchen', 'room'                      # Setting
            ],
            'key_actions': [
                'washing', 'drying', 'dishes',         # Mother's actions
                'stealing', 'taking', 'getting',       # Children's actions
                'climbing', 'falling', 'reaching',     # Risk actions
                'overflowing', 'forgot'               # Water incident
            ],
            'important_objects': [
                'cookies', 'cookie', 'jar',            # Target objects
                'sink', 'water', 'dishes',            # Kitchen items
                'stool', 'chair', 'cabinet',          # Furniture
                'plate'                               # Additional items
            ]
        }

        # Clinical markers for potential cognitive impairment
        self.clinical_markers = {
            'empty_speech': ['thing', 'stuff', 'something'],
            'circumlocutions': ['like', 'kind of', 'sort of'],
            'preservative_terms': ['um', 'uh', 'er', 'ah']
        }

    def analyze_description(self, description):
        doc = self.nlp(description.lower())

        # Clinical analysis metrics
        analysis = {
            'information_units': self._analyze_information_units(doc),
            'linguistic_features': self._analyze_linguistic_features(doc),
            'cognitive_indicators': self._analyze_cognitive_indicators(doc),
            'clinical_assessment': self._generate_clinical_assessment(doc)
        }

        return self._generate_final_report(analysis)

    def _analyze_information_units(self, doc):
        information_units = {
            'subjects_identified': [],
            'actions_described': [],
            'objects_mentioned': []
        }

        text = doc.text.lower()

        for item in self.essential_elements['subject_items']:
            if item in text:
                information_units['subjects_identified'].append(item)

        for action in self.essential_elements['key_actions']:
            if action in text:
                information_units['actions_described'].append(action)

        for obj in self.essential_elements['important_objects']:
            if obj in text:
                information_units['objects_mentioned'].append(obj)

        return information_units

    def _analyze_linguistic_features(self, doc):
        return {
            'sentence_count': len(list(doc.sents)),
            'word_count': len([token for token in doc if not token.is_punct and not token.is_space]),
            'unique_words': len(set([token.text for token in doc if not token.is_punct and not token.is_space])),
            'complex_sentences': len([sent for sent in doc.sents if len(list(sent.root.children)) > 2])
        }

    def _analyze_cognitive_indicators(self, doc):
        text = doc.text.lower()

        return {
            'empty_speech_count': sum(text.count(term) for term in self.clinical_markers['empty_speech']),
            'circumlocutions_count': sum(text.count(term) for term in self.clinical_markers['circumlocutions']),
            'preservative_terms_count': sum(text.count(term) for term in self.clinical_markers['preservative_terms']),
            'repeated_information': self._check_repetition(doc)
        }

    def _check_repetition(self, doc):
        content_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        word_counts = Counter(content_words)
        return sum(1 for count in word_counts.values() if count > 2)  # Only count if repeated more than twice

    def _generate_clinical_assessment(self, doc):
        assessment = {
            'description_completeness': self._assess_completeness(),
            'thematic_coherence': self._assess_coherence(doc),
            'potential_concerns': []
        }
        return assessment

    def _assess_completeness(self):
        total_score = 0
        max_score = 10

        # More lenient scoring
        if len(self.essential_elements['subject_items']) >= 1:
            total_score += 3
        if len(self.essential_elements['key_actions']) >= 1:
            total_score += 4
        if len(self.essential_elements['important_objects']) >= 2:
            total_score += 3

        return (total_score / max_score) * 100

    def _assess_coherence(self, doc):
        sentences = list(doc.sents)
        return {
            'has_introduction': bool(sentences),
            'describes_main_action': any('water' in sent.text.lower() or 'sink' in sent.text.lower() or
                                         'cookie' in sent.text.lower() or 'plate' in sent.text.lower()
                                         for sent in sentences),
            'describes_consequences': any('overflow' in sent.text.lower() or 'forgot' in sent.text.lower() or
                                          'steal' in sent.text.lower() for sent in sentences)
        }

    def _generate_final_report(self, analysis):
        concerns = []
        risk_level = "Low"

        # Information Units Analysis - More lenient thresholds
        total_elements = len(analysis['information_units']['subjects_identified']) + \
                         len(analysis['information_units']['actions_described']) + \
                         len(analysis['information_units']['objects_mentioned'])

        # Adjusted thresholds
        if total_elements < 3:  # Previously 5
            concerns.append("Very limited scene description")
            risk_level = "High"
        elif total_elements < 4:  # Previously 8
            concerns.append("Description could include more elements")
            risk_level = "Medium"

        # Linguistic Features Analysis - More lenient
        if analysis['linguistic_features']['word_count'] < 15:  # Previously 30
            concerns.append("Very brief description")
            risk_level = "Medium" if risk_level != "High" else risk_level

        # Cognitive Indicators Analysis - More lenient
        if analysis['cognitive_indicators']['empty_speech_count'] > 5:  # Previously 3
            concerns.append("Frequent use of non-specific terms")
            risk_level = "High"

        if analysis['cognitive_indicators']['repeated_information'] > 3:  # Previously 2
            concerns.append("Significant information repetition")
            risk_level = "High"

        # Check for key scene elements
        scenes = {
            'water_incident': any('water' in text or 'overflow' in text or 'sink' in text
                                  for text in analysis['information_units']['objects_mentioned'] +
                                  analysis['information_units']['actions_described']),
            'cookie_incident': any('cookie' in text or 'steal' in text
                                   for text in analysis['information_units']['objects_mentioned'] +
                                   analysis['information_units']['actions_described'])
        }

        # Adjust risk level based on scene description
        if scenes['water_incident'] or scenes['cookie_incident']:
            if risk_level == "High":
                risk_level = "Medium"
            elif risk_level == "Medium":
                risk_level = "Low"

        return {
            'risk_level': risk_level,
            'concerns': concerns if concerns else ["No significant concerns identified"],
            'scoring': {
                'information_completeness': total_elements,
                'linguistic_complexity': analysis['linguistic_features']['complex_sentences'],
                'cognitive_markers': sum(analysis['cognitive_indicators'].values())
            },
            'recommendation': self._generate_recommendation(risk_level, total_elements)
        }

    def _generate_recommendation(self, risk_level, total_elements):
        if risk_level == "High":
            return "Consider additional cognitive screening"
        elif risk_level == "Medium":
            return "Monitor and reassess if concerns persist"
        else:
            return "Description appears within normal range"

@main_routes.route('/analyze-picture', methods=['POST'])
def analyze_picture():
    data = request.json
    description = data.get('description', '')

    if not description:
        return jsonify({'error': 'No description provided'}), 400

    try:
        analyzer = CookieTheftAnalyzer()
        analysis = analyzer.analyze_description(description)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

###################################################################################################################


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
