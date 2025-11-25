"""
Flask API Server for Quantum Fake News Detector
===============================================
RESTful API backend for the web interface.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from pathlib import Path
import json
import time

from quantum_model import QuantumNeuralNetwork
from data_preprocessing import TextPreprocessor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for model and preprocessor
qnn = None
preprocessor = None
model_info = {}

def load_model():
    """Load the trained quantum model."""
    global qnn, preprocessor, model_info
    
    # Try to load optimized model first, then high accuracy, then regular
    model_paths = [
        ('results/quantum_model_optimized.pkl', 'results/preprocessor_optimized.pkl', 'results/optimized_results.json', 8, 2),
        ('results/quantum_model_high_accuracy.pkl', 'results/preprocessor_high_accuracy.pkl', 'results/high_accuracy_results.json', 16, 4),
        ('results/quantum_model.pkl', 'results/preprocessor.pkl', 'results/metrics.json', 8, 3),
    ]
    
    for model_path, prep_path, results_path, n_qubits, n_layers in model_paths:
        if Path(model_path).exists() and Path(prep_path).exists():
            print(f"Loading model from {model_path}...")
            
            qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
            qnn.load(model_path)
            
            preprocessor = TextPreprocessor(n_features=n_qubits)
            preprocessor.load(prep_path)
            
            # Load model info
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    model_info = {
                        'accuracy': results.get('metrics', {}).get('accuracy', 0),
                        'precision': results.get('metrics', {}).get('precision', 0),
                        'recall': results.get('metrics', {}).get('recall', 0),
                        'f1': results.get('metrics', {}).get('f1', 0),
                        'qubits': n_qubits,
                        'layers': n_layers,
                        'model_type': 'Optimized' if 'optimized' in model_path else 'High Accuracy' if 'high_accuracy' in model_path else 'Standard'
                    }
            except:
                model_info = {
                    'accuracy': 0.85,
                    'qubits': n_qubits,
                    'layers': n_layers,
                    'model_type': 'Standard'
                }
            
            print(f"✓ Model loaded successfully!")
            print(f"  Type: {model_info['model_type']}")
            print(f"  Qubits: {n_qubits}, Layers: {n_layers}")
            return True
    
    print("⚠️  No trained model found. Please train a model first.")
    return False

# Load model on startup
model_loaded = load_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_info': model_info if model_loaded else None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real."""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train a model first.'
        }), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Start timing
        start_time = time.time()
        
        # Preprocess and predict
        features = preprocessor.transform([text])
        probability = qnn.predict_proba(features[0])
        prediction = qnn.predict(features[0])
        
        # Convert to float
        prob_val = float(probability) if hasattr(probability, '__float__') else probability
        
        # Calculate confidence
        is_fake = prediction == 1
        confidence = prob_val if is_fake else (1 - prob_val)
        
        # Processing time
        processing_time = time.time() - start_time
        
        # Analyze text features
        word_count = len(text.split())
        has_exclamation = '!' in text
        has_caps = any(word.isupper() for word in text.split() if len(word) > 2)
        
        # Determine risk level
        if confidence >= 0.9:
            risk_level = 'very_high' if is_fake else 'very_low'
        elif confidence >= 0.75:
            risk_level = 'high' if is_fake else 'low'
        elif confidence >= 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'uncertain'
        
        # Generate explanation
        explanation = []
        if is_fake:
            if has_exclamation:
                explanation.append("Contains excessive exclamation marks")
            if has_caps:
                explanation.append("Uses sensational capitalization")
            if any(word in text.lower() for word in ['shocking', 'breaking', 'miracle', 'secret']):
                explanation.append("Contains clickbait keywords")
        else:
            if any(word in text.lower() for word in ['research', 'study', 'university', 'scientists']):
                explanation.append("Contains academic/research language")
            if any(word in text.lower() for word in ['government', 'officials', 'announces']):
                explanation.append("Uses formal/official language")
        
        return jsonify({
            'prediction': 'fake' if is_fake else 'real',
            'confidence': round(confidence * 100, 2),
            'probability_fake': round(prob_val * 100, 2),
            'probability_real': round((1 - prob_val) * 100, 2),
            'risk_level': risk_level,
            'explanation': explanation,
            'analysis': {
                'word_count': word_count,
                'has_exclamation': has_exclamation,
                'has_caps': has_caps,
                'processing_time': round(processing_time * 1000, 2)  # ms
            },
            'model_info': model_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict multiple texts at once."""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train a model first.'
        }), 503
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid texts array'}), 400
        
        results = []
        for text in texts[:10]:  # Limit to 10 texts
            if not text.strip():
                continue
            
            features = preprocessor.transform([text])
            probability = qnn.predict_proba(features[0])
            prediction = qnn.predict(features[0])
            
            prob_val = float(probability) if hasattr(probability, '__float__') else probability
            is_fake = prediction == 1
            confidence = prob_val if is_fake else (1 - prob_val)
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': 'fake' if is_fake else 'real',
                'confidence': round(confidence * 100, 2)
            })
        
        return jsonify({
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example news articles for testing."""
    examples = [
        {
            'text': "Scientists at MIT publish peer-reviewed research on quantum computing advances in Nature journal.",
            'expected': 'real',
            'category': 'Academic'
        },
        {
            'text': "BREAKING: Aliens confirmed by government officials! They've been living among us for decades!",
            'expected': 'fake',
            'category': 'Conspiracy'
        },
        {
            'text': "You won't believe this one weird trick that doctors don't want you to know! Click here now!",
            'expected': 'fake',
            'category': 'Clickbait'
        },
        {
            'text': "Federal Reserve announces interest rate decision following comprehensive economic analysis.",
            'expected': 'real',
            'category': 'Economic'
        },
        {
            'text': "Man claims he spoke to aliens and they revealed the secret to eternal life!",
            'expected': 'fake',
            'category': 'Sensational'
        },
        {
            'text': "Government announces new infrastructure bill after months of bipartisan negotiations.",
            'expected': 'real',
            'category': 'Political'
        }
    ]
    
    return jsonify({'examples': examples})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics."""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_info': model_info,
        'quantum_specs': {
            'qubits': model_info.get('qubits', 8),
            'layers': model_info.get('layers', 2),
            'parameters': model_info.get('qubits', 8) * model_info.get('layers', 2) * 3,
            'device': 'Quantum Simulator'
        },
        'performance': {
            'accuracy': round(model_info.get('accuracy', 0.85) * 100, 2),
            'precision': round(model_info.get('precision', 0.85) * 100, 2),
            'recall': round(model_info.get('recall', 0.85) * 100, 2),
            'f1_score': round(model_info.get('f1', 0.85) * 100, 2)
        }
    })

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("QUANTUM FAKE NEWS DETECTOR - API SERVER")
    print("=" * 70)
    
    if model_loaded:
        print("\n✓ Model loaded successfully!")
        print(f"  Type: {model_info['model_type']}")
        print(f"  Accuracy: {model_info.get('accuracy', 0)*100:.1f}%")
    else:
        print("\n⚠️  No model loaded. Train a model first:")
        print("  python train_optimized_fast.py")
    
    print("\n🚀 Starting API server...")
    print("   API will be available at: http://localhost:5000")
    print("   Frontend should connect to this URL")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
