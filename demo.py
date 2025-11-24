"""
Quick Demo Script
================
Demonstrates the quantum fake news detector with pre-trained model or quick training.
"""

import numpy as np
from pathlib import Path

from quantum_model import QuantumNeuralNetwork
from data_preprocessing import TextPreprocessor


def demo_prediction():
    """
    Demo: Make predictions on sample news articles.
    """
    print("=" * 60)
    print("QUANTUM FAKE NEWS DETECTOR - DEMO")
    print("=" * 60)
    
    # Sample news articles to classify
    sample_articles = [
        {
            'text': "Scientists at Stanford University published a peer-reviewed study in Nature journal showing promising results in cancer treatment research.",
            'expected': 'Real'
        },
        {
            'text': "SHOCKING DISCOVERY! Doctors hate this one weird trick that cures everything! Click here now before it's banned!",
            'expected': 'Fake'
        },
        {
            'text': "Government officials announced new infrastructure bill after months of bipartisan negotiations in Congress.",
            'expected': 'Real'
        },
        {
            'text': "BREAKING: Aliens confirmed by NASA! Government has been hiding the truth for decades! Share before deleted!",
            'expected': 'Fake'
        },
        {
            'text': "Economic data released by Federal Reserve shows moderate growth in employment sector during last quarter.",
            'expected': 'Real'
        },
    ]
    
    # Try to load trained model
    model_path = Path('results/quantum_model.pkl')
    preprocessor_path = Path('results/preprocessor.pkl')
    
    if model_path.exists() and preprocessor_path.exists():
        print("\n✓ Loading trained model...")
        
        # Load model and preprocessor
        qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=3)
        qnn.load(str(model_path))
        
        preprocessor = TextPreprocessor(n_features=8)
        preprocessor.load(str(preprocessor_path))
        
        print("✓ Model loaded successfully!")
        
    else:
        print("\n✗ No trained model found. Training a quick demo model...")
        print("(For better results, run: python train.py)")
        
        # Create and train a simple model with synthetic data
        from train import QuantumTrainer
        
        # Generate synthetic training data
        np.random.seed(42)
        n_train = 100
        X_train = np.random.randn(n_train, 8)
        y_train = np.random.randint(0, 2, n_train)
        X_val = np.random.randn(20, 8)
        y_val = np.random.randint(0, 2, 20)
        
        # Quick training
        qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=2)
        trainer = QuantumTrainer(qnn, learning_rate=0.05)
        
        print("\nQuick training (10 epochs)...")
        trainer.train(X_train, y_train, X_val, y_val, epochs=10, verbose=False)
        
        # Create simple preprocessor
        preprocessor = TextPreprocessor(n_features=8)
        # Fit on sample articles
        texts = [article['text'] for article in sample_articles]
        labels = np.array([0, 1, 0, 1, 0])  # Real=0, Fake=1
        preprocessor.fit_transform(texts, labels)
        
        print("✓ Demo model ready!")
    
    # Make predictions
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    
    for i, article in enumerate(sample_articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"Text: {article['text'][:100]}...")
        print(f"Expected: {article['expected']}")
        
        # Preprocess and predict
        features = preprocessor.transform([article['text']])
        probability = qnn.predict_proba(features[0])
        prediction = qnn.predict(features[0])
        
        predicted_label = "Fake" if prediction == 1 else "Real"
        confidence = probability if prediction == 1 else (1 - probability)
        
        print(f"Predicted: {predicted_label} (confidence: {confidence:.2%})")
        
        # Check if correct
        is_correct = predicted_label == article['expected']
        print(f"Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nTo train a full model with real data:")
    print("1. Download a dataset: python download_dataset.py")
    print("2. Train the model: python train.py")
    print("3. Test robustness: python robustness.py")


def interactive_mode():
    """
    Interactive mode: User can input their own text to classify.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    
    # Load model
    model_path = Path('results/quantum_model.pkl')
    preprocessor_path = Path('results/preprocessor.pkl')
    
    if not (model_path.exists() and preprocessor_path.exists()):
        print("\n✗ No trained model found.")
        print("Please run 'python train.py' first to train a model.")
        return
    
    print("\n✓ Loading model...")
    qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=3)
    qnn.load(str(model_path))
    
    preprocessor = TextPreprocessor(n_features=8)
    preprocessor.load(str(preprocessor_path))
    
    print("✓ Model ready!")
    print("\nEnter news articles to classify (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        print("\nEnter article text:")
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        # Make prediction
        try:
            features = preprocessor.transform([text])
            probability = qnn.predict_proba(features[0])
            prediction = qnn.predict(features[0])
            
            predicted_label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
            confidence = probability if prediction == 1 else (1 - probability)
            
            print(f"\n{'='*60}")
            print(f"Prediction: {predicted_label}")
            print(f"Confidence: {confidence:.2%}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error making prediction: {e}")


def main():
    """
    Main demo function.
    """
    print("\n" + "=" * 60)
    print("QUANTUM FAKE NEWS DETECTOR")
    print("=" * 60)
    print("\nChoose demo mode:")
    print("1. Automated demo with sample articles")
    print("2. Interactive mode (enter your own text)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        demo_prediction()
    elif choice == '2':
        interactive_mode()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice. Running automated demo...")
        demo_prediction()


if __name__ == "__main__":
    main()
