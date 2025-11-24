"""
Installation Test Script
=======================
Quick test to verify all components are working correctly.
"""

import sys
from pathlib import Path


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_imports():
    """Test if all required packages can be imported."""
    print_section("Testing Package Imports")
    
    packages = {
        'pennylane': 'PennyLane (Quantum Computing)',
        'numpy': 'NumPy (Numerical Computing)',
        'pandas': 'Pandas (Data Processing)',
        'sklearn': 'Scikit-learn (Machine Learning)',
        'torch': 'PyTorch (Deep Learning)',
        'nltk': 'NLTK (Natural Language Processing)',
        'matplotlib': 'Matplotlib (Visualization)',
        'seaborn': 'Seaborn (Visualization)',
        'tqdm': 'tqdm (Progress Bars)',
    }
    
    results = {}
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
            results[module] = True
        except ImportError as e:
            print(f"✗ {name} - {e}")
            results[module] = False
    
    return all(results.values())


def test_quantum_device():
    """Test PennyLane quantum device."""
    print_section("Testing Quantum Device")
    
    try:
        import pennylane as qml
        import numpy as np
        
        # Create quantum device
        dev = qml.device('default.qubit', wires=2)
        print(f"✓ Created quantum device: {dev}")
        
        # Define simple circuit
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        # Execute circuit
        result = circuit(np.pi / 4)
        print(f"✓ Executed quantum circuit")
        print(f"  Result: {result:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quantum device test failed: {e}")
        return False


def test_preprocessing():
    """Test data preprocessing module."""
    print_section("Testing Data Preprocessing")
    
    try:
        from data_preprocessing import TextPreprocessor
        import numpy as np
        
        # Create preprocessor
        preprocessor = TextPreprocessor(n_features=4)
        print("✓ Created TextPreprocessor")
        
        # Test with sample data (need more samples for TF-IDF)
        texts = [
            "This is a real news article about science and research.",
            "FAKE NEWS! You won't believe this shocking truth!",
            "Government announces new policy on climate change.",
            "Scientists discover breakthrough in quantum computing technology.",
            "Amazing miracle cure that doctors don't want you to know!",
            "Economic report shows steady growth in manufacturing sector.",
            "Breaking news about celebrity scandal and controversy.",
            "Research paper published in Nature reveals important findings.",
        ]
        labels = np.array([0, 1, 0, 0, 1, 0, 1, 0])
        
        # Fit and transform with adjusted parameters for small dataset
        preprocessor.vectorizer = None  # Reset
        preprocessor.pca = None
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import PCA
        
        # Use smaller parameters for test
        preprocessor.vectorizer = TfidfVectorizer(
            max_features=50,  # Smaller for test
            min_df=1,  # Allow all terms
            max_df=1.0,
            ngram_range=(1, 1)
        )
        
        cleaned_texts = [preprocessor.clean_text(text) for text in texts]
        tfidf_features = preprocessor.vectorizer.fit_transform(cleaned_texts).toarray()
        
        # Adjust PCA components based on available features
        n_components = min(4, tfidf_features.shape[1], len(texts))
        preprocessor.pca = PCA(n_components=n_components)
        features = preprocessor.pca.fit_transform(tfidf_features)
        features = preprocessor.scaler.fit_transform(features)
        
        print(f"✓ Preprocessed {len(texts)} texts")
        print(f"  Output shape: {features.shape}")
        print(f"  Feature range: [{features.min():.2f}, {features.max():.2f}]")
        
        # Test transform
        new_features = preprocessor.transform(["Test article"])
        print(f"✓ Transformed new text")
        print(f"  Output shape: {new_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_model():
    """Test quantum model module."""
    print_section("Testing Quantum Model")
    
    try:
        from quantum_model import QuantumNeuralNetwork
        import numpy as np
        
        # Create model
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        print(f"✓ Created QuantumNeuralNetwork")
        print(f"  Qubits: {qnn.n_qubits}")
        print(f"  Layers: {qnn.n_layers}")
        print(f"  Parameters: {qnn.n_params}")
        
        # Test prediction
        features = np.random.randn(4)
        prob = qnn.predict_proba(features)
        pred = qnn.predict(features)
        
        print(f"✓ Made prediction")
        print(f"  Probability: {prob:.4f}")
        print(f"  Class: {pred}")
        
        # Test batch prediction
        X = np.random.randn(5, 4)
        predictions = qnn.predict_batch(X)
        print(f"✓ Batch prediction")
        print(f"  Predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quantum model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of preprocessing and quantum model."""
    print_section("Testing Integration")
    
    try:
        from data_preprocessing import TextPreprocessor
        from quantum_model import QuantumNeuralNetwork
        import numpy as np
        
        # Sample data (need more for TF-IDF)
        texts = [
            "Scientists discover new quantum computing breakthrough in research.",
            "SHOCKING: Aliens confirmed by government officials!",
            "Economic report shows steady growth in technology sector.",
            "You won't believe this miracle cure doctors hate!",
            "Government announces new infrastructure policy for cities.",
            "Amazing secret revealed that will change your life forever!",
            "Research team publishes findings in scientific journal.",
            "Unbelievable conspiracy theory exposed by anonymous source!",
        ]
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Preprocess with adjusted parameters for small test dataset
        print("Preprocessing texts...")
        preprocessor = TextPreprocessor(n_features=4, max_tfidf_features=50)
        
        # Adjust TF-IDF parameters for small dataset
        from sklearn.feature_extraction.text import TfidfVectorizer
        preprocessor.vectorizer = TfidfVectorizer(
            max_features=50,
            min_df=1,  # Allow all terms for small dataset
            max_df=1.0,
            ngram_range=(1, 1)
        )
        
        features, _ = preprocessor.fit_transform(texts, labels)
        print(f"✓ Preprocessed {len(texts)} texts → {features.shape}")
        
        # Create model
        print("Creating quantum model...")
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        print(f"✓ Created model with {qnn.n_params} parameters")
        
        # Make predictions
        print("Making predictions...")
        predictions = qnn.predict_batch(features)
        accuracy = np.mean(predictions == labels)
        
        print(f"✓ Predictions: {predictions}")
        print(f"✓ True labels: {labels}")
        print(f"✓ Random accuracy: {accuracy:.2%}")
        print("  (Note: Untrained model, accuracy should be ~50%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test if all required files exist."""
    print_section("Testing File Structure")
    
    required_files = [
        'data_preprocessing.py',
        'quantum_model.py',
        'train.py',
        'robustness.py',
        'demo.py',
        'config.py',
        'setup.py',
        'download_dataset.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        '.gitignore',
    ]
    
    required_dirs = [
        'data',
        'results',
    ]
    
    all_exist = True
    
    print("\nChecking files:")
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - Missing")
            all_exist = False
    
    print("\nChecking directories:")
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - Missing")
            all_exist = False
    
    return all_exist


def test_nltk_data():
    """Test NLTK data availability."""
    print_section("Testing NLTK Data")
    
    try:
        import nltk
        
        # Try to find required data
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK punkt tokenizer available")
            punkt_ok = True
        except LookupError:
            print("✗ NLTK punkt tokenizer missing")
            print("  Run: python -c 'import nltk; nltk.download(\"punkt\")'")
            punkt_ok = False
        
        try:
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK stopwords available")
            stopwords_ok = True
        except LookupError:
            print("✗ NLTK stopwords missing")
            print("  Run: python -c 'import nltk; nltk.download(\"stopwords\")'")
            stopwords_ok = False
        
        return punkt_ok and stopwords_ok
        
    except Exception as e:
        print(f"✗ NLTK test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("QUANTUM FAKE NEWS DETECTION - INSTALLATION TEST")
    print("=" * 60)
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Package Imports", test_imports),
        ("NLTK Data", test_nltk_data),
        ("Quantum Device", test_quantum_device),
        ("Data Preprocessing", test_preprocessing),
        ("Quantum Model", test_quantum_model),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("Test Summary")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. python download_dataset.py  # Get a dataset")
        print("  2. python train.py             # Train the model")
        print("  3. python demo.py              # Try the demo")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Run: python setup.py")
        print("  - Check Python version (3.8+ required)")
    
    print("\n" + "=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
