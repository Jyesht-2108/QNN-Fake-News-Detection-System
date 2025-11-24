# Quantum Fake News Detection - Project Map 🗺️

## Visual Project Structure

```
quantum-fake-news-detection/
│
├── 📚 DOCUMENTATION
│   ├── README.md                    # Complete project documentation
│   ├── QUICKSTART.md                # 5-minute quick start guide
│   ├── PROJECT_SUMMARY.md           # Technical overview
│   ├── PROJECT_MAP.md               # This file - visual guide
│   └── CHECKLIST.md                 # Completion checklist
│
├── 🔬 CORE MODULES
│   ├── data_preprocessing.py        # Text preprocessing & feature extraction
│   │   ├── TextPreprocessor class
│   │   ├── clean_text()
│   │   ├── fit_transform()
│   │   ├── TF-IDF vectorization
│   │   └── PCA dimensionality reduction
│   │
│   ├── quantum_model.py             # Quantum neural network
│   │   ├── QuantumNeuralNetwork class
│   │   ├── amplitude_encoding()
│   │   ├── angle_encoding()
│   │   ├── variational_layer()
│   │   ├── quantum_circuit()
│   │   └── predict() / predict_batch()
│   │
│   ├── train.py                     # Training pipeline
│   │   ├── QuantumTrainer class
│   │   ├── train_epoch()
│   │   ├── evaluate()
│   │   ├── plot_training_history()
│   │   └── evaluate_model()
│   │
│   └── robustness.py                # Adversarial testing
│       ├── SimpleTextAttacker class
│       ├── RobustnessTester class
│       ├── adversarial_training()
│       └── Multiple attack types
│
├── 🛠️ UTILITIES
│   ├── config.py                    # Configuration management
│   │   ├── Config class
│   │   ├── QuickConfig presets
│   │   └── Validation methods
│   │
│   ├── demo.py                      # Interactive demo
│   │   ├── demo_prediction()
│   │   ├── interactive_mode()
│   │   └── Sample articles
│   │
│   ├── download_dataset.py          # Dataset helper
│   │   ├── download_liar_dataset()
│   │   ├── create_sample_dataset()
│   │   └── check_existing_datasets()
│   │
│   ├── setup.py                     # Installation script
│   │   ├── check_python_version()
│   │   ├── install_dependencies()
│   │   ├── download_nltk_data()
│   │   └── test_quantum_device()
│   │
│   └── test_installation.py         # Verification tests
│       ├── test_imports()
│       ├── test_preprocessing()
│       ├── test_quantum_model()
│       └── test_integration()
│
├── ⚙️ CONFIGURATION
│   ├── requirements.txt             # Python dependencies
│   └── .gitignore                   # Git ignore rules
│
└── 📁 DIRECTORIES
    ├── data/                        # Dataset storage
    │   ├── .gitkeep
    │   └── [Your datasets here]
    │
    ├── results/                     # Output files
    │   ├── quantum_model.pkl
    │   ├── preprocessor.pkl
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── robustness_results.png
    │   └── metrics.json
    │
    └── docs/                        # Additional docs
        └── CONTEXT.md
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    QUANTUM FAKE NEWS DETECTOR                │
└─────────────────────────────────────────────────────────────┘

1. SETUP & INSTALLATION
   ┌──────────────┐
   │  setup.py    │ → Install dependencies
   └──────────────┘ → Download NLTK data
                    → Verify installation

2. DATA ACQUISITION
   ┌──────────────────────┐
   │ download_dataset.py  │ → Download LIAR dataset
   └──────────────────────┘ → Create sample data
                            → Check existing data

3. PREPROCESSING
   ┌──────────────────────────┐
   │ data_preprocessing.py    │
   └──────────────────────────┘
            │
            ├─→ Load dataset (WELFake/LIAR)
            ├─→ Clean text
            ├─→ Tokenize & remove stopwords
            ├─→ TF-IDF vectorization (1000 features)
            ├─→ PCA reduction (8 features)
            └─→ Normalize to [-1, 1]

4. QUANTUM MODEL
   ┌──────────────────┐
   │ quantum_model.py │
   └──────────────────┘
            │
            ├─→ Create quantum device (8 qubits)
            ├─→ Amplitude encoding
            ├─→ Variational layers (3 layers)
            │   ├─→ Parameterized rotations
            │   └─→ Entangling gates
            └─→ Measurement & classification

5. TRAINING
   ┌──────────┐
   │ train.py │
   └──────────┘
            │
            ├─→ Initialize QuantumTrainer
            ├─→ Train for N epochs
            │   ├─→ Forward pass (quantum circuit)
            │   ├─→ Compute loss (cross-entropy)
            │   ├─→ Compute gradients (parameter-shift)
            │   └─→ Update parameters (Adam)
            ├─→ Evaluate on test set
            ├─→ Generate visualizations
            └─→ Save model & metrics

6. EVALUATION
   ┌────────────────────┐
   │ Metrics & Plots    │
   └────────────────────┘
            │
            ├─→ Accuracy, Precision, Recall, F1
            ├─→ Confusion matrix
            ├─→ Training curves
            └─→ Classification report

7. ROBUSTNESS TESTING (Optional)
   ┌────────────────┐
   │ robustness.py  │
   └────────────────┘
            │
            ├─→ Generate adversarial examples
            │   ├─→ Synonym replacement
            │   ├─→ Character swapping
            │   ├─→ Word deletion/insertion
            │   └─→ Mixed attacks
            ├─→ Test model robustness
            ├─→ Compare clean vs. adversarial
            └─→ Visualize results

8. DEPLOYMENT
   ┌──────────┐
   │ demo.py  │
   └──────────┘
            │
            ├─→ Load trained model
            ├─→ Interactive predictions
            └─→ Real-time classification
```

## Data Flow

```
Raw Text
   ↓
[data_preprocessing.py]
   ↓
Clean Text → TF-IDF → PCA → Normalized Features (8D)
   ↓
[quantum_model.py]
   ↓
Amplitude Encoding → Quantum State |ψ⟩
   ↓
Variational Circuit (3 layers)
   ├─→ Layer 1: Rotations + Entanglement
   ├─→ Layer 2: Rotations + Entanglement
   └─→ Layer 3: Rotations + Entanglement
   ↓
Measurement ⟨Z₀⟩
   ↓
Expectation Value [-1, 1]
   ↓
Probability [0, 1]
   ↓
Classification: Real (0) or Fake (1)
```

## Module Dependencies

```
┌─────────────────────┐
│   External Libs     │
│  - PennyLane        │
│  - NumPy            │
│  - Pandas           │
│  - Scikit-learn     │
│  - NLTK             │
│  - Matplotlib       │
└─────────────────────┘
          ↓
┌─────────────────────┐
│  data_preprocessing │ ←─────┐
└─────────────────────┘       │
          ↓                   │
┌─────────────────────┐       │
│   quantum_model     │       │
└─────────────────────┘       │
          ↓                   │
┌─────────────────────┐       │
│      train.py       │───────┤
└─────────────────────┘       │
          ↓                   │
┌─────────────────────┐       │
│   robustness.py     │───────┘
└─────────────────────┘
          ↓
┌─────────────────────┐
│      demo.py        │
└─────────────────────┘
```

## Quick Command Reference

### Installation & Setup
```bash
python setup.py                    # Full installation
python test_installation.py        # Verify installation
```

### Data Management
```bash
python download_dataset.py         # Get dataset
# Option 1: Download LIAR
# Option 2: Create sample data
# Option 3: Manual WELFake instructions
```

### Training & Evaluation
```bash
python train.py                    # Train model (main pipeline)
python robustness.py               # Test adversarial robustness
python demo.py                     # Interactive demo
```

### Testing Individual Modules
```bash
python data_preprocessing.py       # Test preprocessing
python quantum_model.py            # Test quantum model
python config.py                   # Test configuration
```

## Configuration Quick Reference

### Fast Testing (5 minutes)
```python
SAMPLE_SIZE = 200
N_FEATURES = 4
N_QUBITS = 4
N_LAYERS = 2
EPOCHS = 20
```

### Standard Training (20 minutes)
```python
SAMPLE_SIZE = 1000
N_FEATURES = 8
N_QUBITS = 8
N_LAYERS = 3
EPOCHS = 50
```

### Full Training (60 minutes)
```python
SAMPLE_SIZE = None  # All data
N_FEATURES = 8
N_QUBITS = 8
N_LAYERS = 3
EPOCHS = 100
```

## Output Files Reference

### After Training
```
results/
├── quantum_model.pkl           # Trained quantum model
├── preprocessor.pkl            # Fitted text preprocessor
├── training_history.png        # Loss & accuracy curves
├── confusion_matrix.png        # Test set confusion matrix
└── metrics.json                # Evaluation metrics
```

### After Robustness Testing
```
results/
├── robustness_results.png      # Attack comparison plot
└── robustness_metrics.json     # Robustness metrics
```

## Key Classes & Functions

### TextPreprocessor
```python
preprocessor = TextPreprocessor(n_features=8)
features, labels = preprocessor.fit_transform(texts, labels)
preprocessor.save('preprocessor.pkl')
```

### QuantumNeuralNetwork
```python
qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=3)
prediction = qnn.predict(features)
qnn.save('model.pkl')
```

### QuantumTrainer
```python
trainer = QuantumTrainer(qnn, learning_rate=0.01)
trainer.train(X_train, y_train, X_val, y_val, epochs=50)
trainer.plot_training_history()
```

### RobustnessTester
```python
tester = RobustnessTester(qnn, preprocessor)
results = tester.test_robustness(texts, labels)
tester.plot_robustness_results(results)
```

## Learning Path

### Beginner
1. Read QUICKSTART.md
2. Run setup.py
3. Try demo.py
4. Read inline comments in quantum_model.py

### Intermediate
1. Read README.md
2. Train with sample data
3. Modify config.py
4. Experiment with hyperparameters

### Advanced
1. Read PROJECT_SUMMARY.md
2. Train on full dataset
3. Implement custom quantum circuits
4. Add new attack types
5. Compare with classical models

## Troubleshooting Map

```
Problem: Installation fails
   → Check Python version (3.8+)
   → Run: pip install -r requirements.txt
   → Check internet connection

Problem: Dataset not found
   → Run: python download_dataset.py
   → Or code will use synthetic data

Problem: Training too slow
   → Reduce SAMPLE_SIZE in train.py
   → Reduce N_QUBITS and N_LAYERS
   → Reduce EPOCHS

Problem: Out of memory
   → Reduce BATCH_SIZE
   → Reduce SAMPLE_SIZE
   → Close other applications

Problem: Poor accuracy
   → Increase EPOCHS
   → Increase N_LAYERS
   → Use more training data
   → Adjust LEARNING_RATE
```

## Success Metrics

✅ Installation completes without errors  
✅ All tests pass in test_installation.py  
✅ Training runs and converges  
✅ Plots are generated  
✅ Model achieves >60% accuracy  
✅ Demo works with predictions  

## Next Steps After Setup

1. ✅ Verify installation: `python test_installation.py`
2. ✅ Get dataset: `python download_dataset.py`
3. ✅ Quick test: Edit train.py → Set SAMPLE_SIZE=200, EPOCHS=20
4. ✅ Run training: `python train.py`
5. ✅ Check results: Open results/training_history.png
6. ✅ Try demo: `python demo.py`
7. ✅ Test robustness: `python robustness.py`
8. ✅ Experiment: Modify config.py and retrain

---

**This map provides a complete visual overview of the project structure, workflow, and usage patterns.**

For detailed information, refer to:
- **Quick Start**: QUICKSTART.md
- **Full Documentation**: README.md
- **Technical Details**: PROJECT_SUMMARY.md
- **Completion Status**: CHECKLIST.md
