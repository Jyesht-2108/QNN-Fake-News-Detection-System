"""
Demo for Optimized Fast Model
=============================
Test the fast-trained model.
"""

import numpy as np
from pathlib import Path
from quantum_model import QuantumNeuralNetwork
from data_preprocessing import TextPreprocessor


print("=" * 70)
print("OPTIMIZED QUANTUM FAKE NEWS DETECTOR - DEMO")
print("=" * 70)

# Load model
model_path = Path('results/quantum_model_optimized.pkl')
preprocessor_path = Path('results/preprocessor_optimized.pkl')

if not model_path.exists():
    print("\n❌ Model not found!")
    print("Please train first: python train_optimized_fast.py")
    exit(1)

print("\n📥 Loading model...")
qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=2)
qnn.load(str(model_path))

preprocessor = TextPreprocessor(n_features=8)
preprocessor.load(str(preprocessor_path))

print("✓ Model loaded!")

# Load results
import json
try:
    with open('results/optimized_results.json', 'r') as f:
        results = json.load(f)
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy:  {results['metrics']['accuracy']:.2%}")
    print(f"  Precision: {results['metrics']['precision']:.2%}")
    print(f"  F1-Score:  {results['metrics']['f1']:.2%}")
except:
    pass

# Test cases
print("\n" + "=" * 70)
print("TESTING WITH YOUR EXAMPLES")
print("=" * 70)

test_cases = [
    ("Man claims he spoke to aliens!!", "FAKE"),
    ("Scientists at MIT publish peer-reviewed research", "REAL"),
    ("You won't believe this miracle cure!", "FAKE"),
    ("Government announces new policy", "REAL"),
    ("SHOCKING: Time traveler from future warns us!", "FAKE"),
]

correct = 0
for text, expected in test_cases:
    features = preprocessor.transform([text])
    prob = qnn.predict_proba(features[0])
    pred = qnn.predict(features[0])
    
    prob_val = float(prob) if hasattr(prob, '__float__') else prob
    predicted = "FAKE" if pred == 1 else "REAL"
    conf = prob_val if pred == 1 else (1 - prob_val)
    
    is_correct = predicted == expected
    if is_correct:
        correct += 1
    
    print(f"\n{'─' * 70}")
    print(f"Text: \"{text}\"")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted} ({conf:.1%} confidence)")
    print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

print(f"\n{'=' * 70}")
print(f"Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")
print("=" * 70)

# Interactive mode
print("\n" + "=" * 70)
print("INTERACTIVE MODE")
print("=" * 70)
print("\nEnter news text to classify (or 'quit' to exit)")

while True:
    print("\n" + "─" * 70)
    user_input = input("Enter text: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    features = preprocessor.transform([user_input])
    prob = qnn.predict_proba(features[0])
    pred = qnn.predict(features[0])
    
    prob_val = float(prob) if hasattr(prob, '__float__') else prob
    predicted = "FAKE NEWS" if pred == 1 else "REAL NEWS"
    conf = prob_val if pred == 1 else (1 - prob_val)
    
    print(f"\n{'═' * 70}")
    print(f"Prediction: {predicted}")
    print(f"Confidence: {conf:.1%}")
    print(f"{'═' * 70}")

print("\nExiting...")
