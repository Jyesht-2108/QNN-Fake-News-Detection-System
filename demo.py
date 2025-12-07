"""
Demo Script for Hybrid Classical-Quantum Fake News Detector
============================================================
Load trained model and make predictions on new text.
"""

import torch
import numpy as np
import sys
from pathlib import Path

from quantum_model import HybridQuantumModel
from data_preprocessing import BERTPCAPreprocessor


# Configuration (must match training)
N_QUBITS = 4
N_LAYERS = 3
N_FEATURES = 8

print("=" * 60)
print("   HYBRID QUANTUM FAKE NEWS DETECTOR - DEMO")
print("=" * 60)

# Check if model files exist
if not Path('results/quantum_model.pth').exists():
    print("\nERROR: Model file not found at results/quantum_model.pth")
    print("Please run train.py first to train the model.")
    sys.exit(1)

if not Path('results/preprocessor.pkl').exists():
    print("\nERROR: Preprocessor file not found at results/preprocessor.pkl")
    print("Please run train.py first to train the model.")
    sys.exit(1)

# Load preprocessor
print("\nLoading preprocessor...")
preprocessor = BERTPCAPreprocessor(n_components=N_FEATURES)
preprocessor.load('results/preprocessor.pkl')
print("✓ Preprocessor loaded")

# Load model
print("\nLoading quantum model...")
model = HybridQuantumModel(n_qubits=N_QUBITS, n_layers=N_LAYERS)
model.load('results/quantum_model.pth')
model.eval()
print("✓ Model loaded")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✓ Using device: {device}")


def predict_news(text: str):
    """
    Predict if a news article is real or fake.
    
    Args:
        text: News article text
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: '{text[:100]}...'")
    print(f"{'='*60}")
    
    try:
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
        
        # Display result
        if label == "FAKE":
            print(f"\n🚨 Prediction: {label} NEWS 🚨")
        else:
            print(f"\n✅ Prediction: {label} NEWS ✅")
        
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Probability (Fake): {prob:.4f}")
        
    except Exception as e:
        print(f"\nERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE NEWS ARTICLES")
    print("="*60)
    
    # Test cases
    test_articles = [
        "Suspected toxic gas leak in Dhanbad kills two, several hospitalized",
        "Scientists discover water on Mars, confirming potential for life",
        "BREAKING: Pope Francis endorses Donald Trump for President!",
        "Government passes new tax bill affecting small businesses",
        "You won't believe what this celebrity did! Doctors hate this one trick!",
        "Climate change report shows rising global temperatures over past decade"
    ]
    
    for article in test_articles:
        predict_news(article)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
