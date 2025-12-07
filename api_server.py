"""
Flask API Server for Hybrid Quantum Fake News Detector
======================================================
Provides REST API endpoints for fake news detection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from pathlib import Path
import sys

from quantum_model import HybridQuantumModel
from data_preprocessing import BERTPCAPreprocessor


# Configuration
N_QUBITS = 4
N_LAYERS = 3
N_FEATURES = 8
MODEL_PATH = 'results/quantum_model.pth'
PREPROCESSOR_PATH = 'results/preprocessor.pkl'

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None
device = None


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor, device
    
    print("Loading model and preprocessor...")
    
    # Check if files exist
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run train.py first.")
        sys.exit(1)
    
    if not Path(PREPROCESSOR_PATH).exists():
        print(f"ERROR: Preprocessor not found at {PREPROCESSOR_PATH}")
        print("Please run train.py first.")
        sys.exit(1)
    
    # Load preprocessor
    preprocessor = BERTPCAPreprocessor(n_components=N_FEATURES)
    preprocessor.load(PREPROCESSOR_PATH)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridQuantumModel(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    model.load(MODEL_PATH)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict if a news article is fake or real.
    
    Request body:
    {
        "text": "News article text here..."
    }
    
    Response:
    {
        "prediction": "FAKE" or "REAL",
        "probability": 0.85,
        "confidence": 0.85
    }
    """
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request body'
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Preprocess text
        features = preprocessor.transform([text])
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(device)
        
        # Get prediction
        with torch.no_grad():
            prob = model(features_tensor).item()
        
        # Interpret result
        label = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if label == "FAKE" else (1 - prob)
        
        return jsonify({
            'prediction': label,
            'probability': float(prob),
            'confidence': float(confidence),
            'text_preview': text[:100] + '...' if len(text) > 100 else text
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple news articles at once.
    
    Request body:
    {
        "texts": ["Article 1...", "Article 2...", ...]
    }
    
    Response:
    {
        "predictions": [
            {"prediction": "FAKE", "probability": 0.85, "confidence": 0.85},
            {"prediction": "REAL", "probability": 0.25, "confidence": 0.75},
            ...
        ]
    }
    """
    try:
        # Get texts from request
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing "texts" field in request body'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'error': 'texts must be a non-empty list'
            }), 400
        
        # Preprocess all texts
        features = preprocessor.transform(texts)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(device)
        
        # Get predictions
        with torch.no_grad():
            probs = model(features_tensor).cpu().numpy().flatten()
        
        # Format results
        results = []
        for prob in probs:
            label = "FAKE" if prob > 0.5 else "REAL"
            confidence = prob if label == "FAKE" else (1 - prob)
            results.append({
                'prediction': label,
                'probability': float(prob),
                'confidence': float(confidence)
            })
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
@app.route('/api/model_info', methods=['GET'])
@app.route('/api/stats', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    return jsonify({
        'n_qubits': N_QUBITS,
        'n_layers': N_LAYERS,
        'n_features': N_FEATURES,
        'device': str(device),
        'model_path': MODEL_PATH,
        'preprocessor_path': PREPROCESSOR_PATH,
        'total_predictions': 0,
        'accuracy': 0.0
    })


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example news articles for testing."""
    examples = [
        {
            'id': 1,
            'text': 'Suspected toxic gas leak in Dhanbad kills two, several hospitalized',
            'label': 'Real News'
        },
        {
            'id': 2,
            'text': 'Scientists discover water on Mars, confirming potential for life',
            'label': 'Real News'
        },
        {
            'id': 3,
            'text': 'BREAKING: Pope Francis endorses Donald Trump for President!',
            'label': 'Fake News'
        },
        {
            'id': 4,
            'text': 'Government passes new tax bill affecting small businesses',
            'label': 'Real News'
        },
        {
            'id': 5,
            'text': 'You won\'t believe what this celebrity did! Doctors hate this one trick!',
            'label': 'Fake News'
        }
    ]
    return jsonify({'examples': examples})


if __name__ == '__main__':
    print("=" * 60)
    print("HYBRID QUANTUM FAKE NEWS DETECTOR - API SERVER")
    print("=" * 60)
    
    # Load model and preprocessor
    load_model_and_preprocessor()
    
    print("\nStarting Flask server...")
    print("API Endpoints:")
    print("  GET  /health          - Health check")
    print("  POST /predict         - Predict single article")
    print("  POST /batch_predict   - Predict multiple articles")
    print("  GET  /model_info      - Model information")
    print("\n" + "=" * 60)
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)
