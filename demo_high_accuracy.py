"""
High-Accuracy Model Demo
========================
Test your 90%+ accuracy quantum model with adversarial robustness.
"""

import numpy as np
from pathlib import Path

from quantum_model import QuantumNeuralNetwork
from data_preprocessing import TextPreprocessor


def test_model():
    """Test the high-accuracy model."""
    print("=" * 70)
    print("HIGH-ACCURACY QUANTUM FAKE NEWS DETECTOR - DEMO")
    print("=" * 70)
    
    # Load model
    model_path = Path('results/quantum_model_high_accuracy.pkl')
    preprocessor_path = Path('results/preprocessor_high_accuracy.pkl')
    
    if not model_path.exists():
        print("\n❌ High-accuracy model not found!")
        print("Please train the model first:")
        print("  python train_high_accuracy.py")
        return
    
    print("\n📥 Loading high-accuracy model...")
    qnn = QuantumNeuralNetwork(n_qubits=16, n_layers=4)
    qnn.load(str(model_path))
    
    preprocessor = TextPreprocessor(n_features=16)
    preprocessor.load(str(preprocessor_path))
    
    print("✓ Model loaded successfully!")
    
    # Load results
    import json
    try:
        with open('results/high_accuracy_results.json', 'r') as f:
            results = json.load(f)
        print(f"\n📊 Model Performance:")
        print(f"  Accuracy:  {results['metrics']['accuracy']:.2%}")
        print(f"  Precision: {results['metrics']['precision']:.2%}")
        print(f"  Recall:    {results['metrics']['recall']:.2%}")
        print(f"  F1-Score:  {results['metrics']['f1']:.2%}")
    except:
        pass
    
    # Test cases including your example
    print("\n" + "=" * 70)
    print("TESTING WITH DIVERSE EXAMPLES")
    print("=" * 70)
    
    test_cases = [
        {
            'text': "Man claims he spoke to aliens!!",
            'expected': 'FAKE',
            'category': 'Sensational claim'
        },
        {
            'text': "BREAKING: Scientists confirm aliens living among us, government admits!",
            'expected': 'FAKE',
            'category': 'Conspiracy theory'
        },
        {
            'text': "You won't believe this miracle cure that doctors don't want you to know!",
            'expected': 'FAKE',
            'category': 'Clickbait medical'
        },
        {
            'text': "Scientists at MIT publish peer-reviewed research on quantum computing advances.",
            'expected': 'REAL',
            'category': 'Academic news'
        },
        {
            'text': "Government announces new infrastructure bill after bipartisan negotiations.",
            'expected': 'REAL',
            'category': 'Political news'
        },
        {
            'text': "SHOCKING revelation: Moon landing was completely faked in Hollywood studio!",
            'expected': 'FAKE',
            'category': 'Conspiracy theory'
        },
        {
            'text': "Federal Reserve announces interest rate decision following economic analysis.",
            'expected': 'REAL',
            'category': 'Economic news'
        },
        {
            'text': "Time traveler from 2050 warns about upcoming disaster next week!",
            'expected': 'FAKE',
            'category': 'Impossible claim'
        },
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}/{total}: {case['category']}")
        print(f"{'─' * 70}")
        print(f"Text: \"{case['text']}\"")
        print(f"Expected: {case['expected']}")
        
        # Make prediction
        features = preprocessor.transform([case['text']])
        probability = qnn.predict_proba(features[0])
        prediction = qnn.predict(features[0])
        
        # Convert probability to float for display
        prob_val = float(probability) if hasattr(probability, '__float__') else probability
        
        predicted_label = "FAKE" if prediction == 1 else "REAL"
        confidence = prob_val if prediction == 1 else (1 - prob_val)
        
        print(f"Predicted: {predicted_label} (confidence: {confidence:.1%})")
        
        is_correct = predicted_label == case['expected']
        if is_correct:
            print("Result: ✅ CORRECT")
            correct += 1
        else:
            print("Result: ❌ INCORRECT")
    
    # Summary
    accuracy = correct / total
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.875:  # 7/8 or better
        print("\n✅ Excellent performance on diverse test cases!")
    elif accuracy >= 0.75:
        print("\n⚠️  Good performance, but could be improved.")
    else:
        print("\n❌ Model needs more training or better data.")
    
    # Test adversarial robustness
    print("\n" + "=" * 70)
    print("TESTING ADVERSARIAL ROBUSTNESS")
    print("=" * 70)
    
    test_text = "Man claims he spoke to aliens!!"
    variations = [
        "Man claims he spoke to aliens!!",
        "Man says he talked to extraterrestrials!!",
        "Person alleges communication with alien beings!!",
        "Individual reports conversation with space aliens!!",
        "Guy states he communicated with aliens from space!!",
    ]
    
    print(f"\nOriginal: \"{test_text}\"")
    print("\nTesting variations:")
    
    predictions = []
    for i, var in enumerate(variations, 1):
        features = preprocessor.transform([var])
        pred = qnn.predict(features[0])
        prob = qnn.predict_proba(features[0])
        predictions.append(pred)
        
        # Convert to float for display
        prob_val = float(prob) if hasattr(prob, '__float__') else prob
        
        label = "FAKE" if pred == 1 else "REAL"
        conf = prob_val if pred == 1 else (1 - prob_val)
        print(f"  {i}. {label} ({conf:.1%}) - \"{var}\"")
    
    # Check consistency
    consistency = len(set(predictions)) == 1
    if consistency:
        print(f"\n✅ Robust! All variations classified as: {'FAKE' if predictions[0] == 1 else 'REAL'}")
    else:
        print(f"\n⚠️  Inconsistent predictions across variations")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nEnter your own news text to classify (or 'quit' to exit)")
    
    while True:
        print("\n" + "─" * 70)
        user_input = input("Enter news text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            continue
        
        # Predict
        features = preprocessor.transform([user_input])
        probability = qnn.predict_proba(features[0])
        prediction = qnn.predict(features[0])
        
        # Convert to float for display
        prob_val = float(probability) if hasattr(probability, '__float__') else probability
        
        predicted_label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        confidence = prob_val if prediction == 1 else (1 - prob_val)
        
        print(f"\n{'═' * 70}")
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.1%}")
        print(f"{'═' * 70}")
        
        # Provide explanation
        if prediction == 1:
            print("\n🚨 This appears to be FAKE NEWS because:")
            if any(word in user_input.lower() for word in ['shocking', 'breaking', 'won\'t believe', 'miracle']):
                print("  • Contains sensational language")
            if '!' in user_input and user_input.count('!') > 1:
                print("  • Excessive exclamation marks")
            if any(word in user_input.lower() for word in ['aliens', 'conspiracy', 'secret', 'exposed']):
                print("  • Contains conspiracy-related keywords")
        else:
            print("\n✅ This appears to be REAL NEWS because:")
            if any(word in user_input.lower() for word in ['research', 'study', 'university', 'scientists']):
                print("  • Contains academic/research language")
            if any(word in user_input.lower() for word in ['government', 'announces', 'officials']):
                print("  • Contains official/formal language")
            if user_input.count('!') == 0:
                print("  • Uses neutral, non-sensational tone")


if __name__ == "__main__":
    test_model()
