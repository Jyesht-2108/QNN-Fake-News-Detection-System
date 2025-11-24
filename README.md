# Quantum Neural Network for Fake News Detection 🔬🗞️

A complete implementation of a quantum neural network (QNN) using PennyLane to detect fake news. This project demonstrates how quantum computing can be applied to natural language processing tasks with adversarial robustness testing.

## 📋 Project Overview

This project implements a **Variational Quantum Classifier (VQC)** that combines classical text preprocessing with quantum machine learning to classify news articles as real or fake. The system includes:

- **Classical preprocessing**: Text cleaning, TF-IDF feature extraction, and PCA dimensionality reduction
- **Quantum encoding**: Amplitude encoding to map classical features to quantum states
- **Variational quantum circuit**: Parameterized quantum layers with trainable weights
- **Hybrid training**: Classical optimizer (Adam) training quantum parameters
- **Adversarial testing**: Robustness evaluation against text perturbations

## 🎯 Features

✅ Beginner-friendly, well-commented code  
✅ Modular architecture with separate preprocessing, model, and training modules  
✅ Support for WELFake and LIAR datasets  
✅ Complete training pipeline with visualization  
✅ Adversarial robustness testing  
✅ Model persistence (save/load)  
✅ Comprehensive evaluation metrics  
✅ Training curves and confusion matrix plots  

## 📁 Project Structure

```
quantum-fake-news-detection/
├── data_preprocessing.py      # Text preprocessing and feature extraction
├── quantum_model.py           # Quantum circuit and VQC implementation
├── train.py                   # Training and evaluation pipeline
├── robustness.py              # Adversarial robustness testing
├── download_dataset.py        # Helper script to download datasets
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Dataset directory (create this)
│   └── WELFake_Dataset.csv   # Place dataset here
└── results/                   # Output directory (auto-created)
    ├── quantum_model.pkl
    ├── preprocessor.pkl
    ├── training_history.png
    ├── confusion_matrix.png
    └── metrics.json
```

## 🚀 Quick Start

### 1. Installation

First, clone or download this project, then install dependencies:

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: Installation may take a few minutes as it includes PennyLane, PyTorch, and other ML libraries.

### 2. Download Dataset

You have two options:

**Option A: WELFake Dataset (Recommended)**
1. Download from: https://mldata.vn/english/welfake
2. Extract and place `WELFake_Dataset.csv` in the `data/` directory

**Option B: LIAR Dataset**
1. Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
2. Extract and place the TSV files in the `data/` directory

**Option C: Use Synthetic Data**
- If no dataset is available, the code will automatically generate synthetic data for demonstration

### 3. Test Individual Modules

Test each module independently to ensure everything works:

```bash
# Test preprocessing
python data_preprocessing.py

# Test quantum model
python quantum_model.py
```

### 4. Train the Model

Run the complete training pipeline:

```bash
python train.py
```

**Training Configuration** (edit in `train.py`):
- `DATASET_PATH`: Path to your dataset
- `N_FEATURES`: Number of quantum features (default: 8)
- `N_QUBITS`: Number of qubits (default: 8)
- `N_LAYERS`: Variational layers (default: 3)
- `EPOCHS`: Training epochs (default: 50)
- `SAMPLE_SIZE`: Limit samples for faster testing (default: 1000, set to None for full dataset)

**Expected Output**:
- Training progress with loss and accuracy
- Training history plot (`results/training_history.png`)
- Confusion matrix (`results/confusion_matrix.png`)
- Evaluation metrics (`results/metrics.json`)
- Saved model (`results/quantum_model.pkl`)

### 5. Test Adversarial Robustness (Optional)

Test how robust your model is against adversarial text attacks:

```bash
python robustness.py
```

This will:
- Load your trained model
- Generate adversarial examples using various attack strategies
- Evaluate model performance on perturbed inputs
- Create robustness visualization (`results/robustness_results.png`)

## 📊 Understanding the Results

### Training Metrics

After training, you'll see:

```
Test Set Performance:
  Accuracy:  0.8500
  Precision: 0.8300
  Recall:    0.8700
  F1-Score:  0.8495
```

- **Accuracy**: Overall correctness
- **Precision**: Of predicted fake news, how many were actually fake
- **Recall**: Of actual fake news, how many were detected
- **F1-Score**: Harmonic mean of precision and recall

### Visualizations

1. **Training History** (`training_history.png`):
   - Left plot: Training and validation loss over epochs
   - Right plot: Training and validation accuracy over epochs
   - Look for convergence and avoid overfitting

2. **Confusion Matrix** (`confusion_matrix.png`):
   - Shows true positives, false positives, true negatives, false negatives
   - Diagonal elements = correct predictions

3. **Robustness Results** (`robustness_results.png`):
   - Compares accuracy on clean vs. adversarial examples
   - Shows model vulnerability to different attack types

## 🔬 How It Works

### 1. Text Preprocessing

```python
Text → Clean → Tokenize → TF-IDF → PCA → Normalized Features
```

- Removes URLs, punctuation, stopwords
- Extracts TF-IDF features (captures word importance)
- Reduces to 8 dimensions using PCA
- Normalizes to [-1, 1] range for quantum encoding

### 2. Quantum Encoding

Classical features are encoded into quantum states using **amplitude encoding**:

```
Classical vector [x₁, x₂, ..., x₈] → Quantum state |ψ⟩ = Σᵢ xᵢ|i⟩
```

This maps 8 classical features to the amplitudes of an 8-qubit quantum state.

### 3. Variational Quantum Circuit

The quantum circuit consists of:

1. **Encoding layer**: Amplitude embedding of input features
2. **Variational layers** (repeated N times):
   - Parameterized rotations (RX, RY, RZ) on each qubit
   - Entangling CNOT gates between qubits
3. **Measurement**: Expectation value of Pauli-Z on first qubit

```
|0⟩ ──[Encoding]──[Rot]──●──[Rot]──●──[Measure]
|0⟩ ──[Encoding]──[Rot]──X──[Rot]──X──
|0⟩ ──[Encoding]──[Rot]──●──[Rot]──●──
...
```

### 4. Training

- **Optimizer**: Adam (classical)
- **Loss**: Binary cross-entropy
- **Gradient**: Computed using parameter-shift rule (quantum-specific)
- **Batch training**: Processes samples in batches for efficiency

### 5. Classification

The measurement output (expectation value) is converted to probability:

```
Expectation ∈ [-1, 1] → Probability ∈ [0, 1]
Probability ≥ 0.5 → Fake News (class 1)
Probability < 0.5 → Real News (class 0)
```

## 🛠️ Customization

### Adjust Quantum Circuit

In `quantum_model.py`, modify:

```python
# Change number of qubits
qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)

# Use angle encoding instead of amplitude encoding
def quantum_circuit(self, features, params):
    self.angle_encoding(features)  # Instead of amplitude_encoding
    ...
```

### Change Preprocessing

In `data_preprocessing.py`:

```python
# Adjust feature dimensions
preprocessor = TextPreprocessor(
    n_features=4,              # Fewer features
    max_tfidf_features=500     # Smaller vocabulary
)

# Modify text cleaning
def clean_text(self, text):
    # Add custom cleaning logic
    ...
```

### Training Hyperparameters

In `train.py`:

```python
# Experiment with different settings
EPOCHS = 100              # More training
LEARNING_RATE = 0.001     # Smaller learning rate
BATCH_SIZE = 20           # Larger batches
N_LAYERS = 4              # Deeper circuit
```

## 🧪 Adversarial Robustness

The `robustness.py` module tests model resilience against:

1. **Synonym Replacement**: Replace words with synonyms
2. **Character Swapping**: Swap adjacent characters (typos)
3. **Word Deletion**: Randomly remove words
4. **Word Insertion**: Add duplicate words
5. **Mixed Attacks**: Combination of above

### Adversarial Training

To improve robustness, train with adversarial examples:

```python
from robustness import adversarial_training

# Train with 30% adversarial examples
qnn_robust = adversarial_training(
    qnn, preprocessor,
    texts_train, labels_train,
    texts_val, labels_val,
    augmentation_ratio=0.3
)
```

## 📈 Performance Tips

### For Faster Training

1. **Reduce sample size**: Set `SAMPLE_SIZE = 500` in `train.py`
2. **Fewer features**: Use `N_FEATURES = 4` instead of 8
3. **Fewer layers**: Use `N_LAYERS = 2` instead of 3
4. **Fewer epochs**: Start with `EPOCHS = 20`

### For Better Accuracy

1. **More data**: Use full dataset (`SAMPLE_SIZE = None`)
2. **More features**: Try `N_FEATURES = 16` (requires 16 qubits)
3. **Deeper circuit**: Increase `N_LAYERS = 4`
4. **Longer training**: Use `EPOCHS = 100`
5. **Better features**: Use BERT embeddings instead of TF-IDF

## 🐛 Troubleshooting

### Issue: "Dataset not found"

**Solution**: Download the dataset and place it in the `data/` directory, or let the code use synthetic data for testing.

### Issue: "Out of memory"

**Solution**: Reduce `SAMPLE_SIZE`, `N_FEATURES`, or `BATCH_SIZE` in `train.py`.

### Issue: "Training is very slow"

**Solution**: 
- Quantum simulation is computationally expensive
- Reduce `N_QUBITS`, `N_LAYERS`, or `SAMPLE_SIZE`
- Use a smaller `BATCH_SIZE`
- Consider using GPU acceleration (requires additional setup)

### Issue: "Poor accuracy"

**Solution**:
- Quantum models may need more epochs to converge
- Try different learning rates (0.001 - 0.1)
- Increase `N_LAYERS` for more expressiveness
- Ensure dataset is balanced (equal real/fake samples)

### Issue: "NLTK data not found"

**Solution**: The code auto-downloads NLTK data, but you can manually run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📚 Learn More

### Quantum Machine Learning
- [PennyLane Documentation](https://pennylane.ai/)
- [Quantum Machine Learning Tutorial](https://pennylane.ai/qml/)
- [Variational Quantum Classifiers](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html)

### Fake News Detection
- [WELFake Dataset Paper](https://arxiv.org/abs/2201.13367)
- [LIAR Dataset Paper](https://arxiv.org/abs/1705.00648)

### Adversarial Robustness
- [TextAttack Library](https://github.com/QData/TextAttack)
- [Adversarial NLP](https://arxiv.org/abs/1801.07175)

## 🤝 Contributing

This is a beginner-friendly educational project. Feel free to:
- Experiment with different quantum circuits
- Try other datasets
- Implement additional attack strategies
- Optimize performance
- Add new features

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_fake_news_detector,
  title = {Quantum Neural Network for Fake News Detection},
  author = {Your Name},
  year = {2024},
  description = {A PennyLane-based quantum classifier for fake news detection}
}
```

## ⚖️ License

This project is provided for educational purposes. Please ensure you comply with the licenses of the datasets you use (WELFake, LIAR).

## 🎓 Educational Notes

### Why Quantum for Fake News Detection?

1. **Feature Space**: Quantum states can represent exponentially large feature spaces
2. **Entanglement**: Captures complex correlations between features
3. **Research**: Explores quantum advantage in NLP tasks
4. **Learning**: Great introduction to quantum machine learning

### Limitations

- **Simulation**: Running on classical computers (slow)
- **Noise**: Real quantum hardware has errors (not addressed here)
- **Scalability**: Limited by number of qubits available
- **Advantage**: Quantum advantage not yet proven for this task

### Next Steps

1. **Try real quantum hardware**: Use IBM Quantum or other cloud services
2. **Implement noise models**: Add realistic quantum noise
3. **Compare with classical**: Benchmark against classical neural networks
4. **Explore other encodings**: Try different quantum feature maps
5. **Scale up**: Test with larger datasets and more qubits

---

**Happy Quantum Computing! 🚀**

For questions or issues, please check the troubleshooting section or review the inline code comments.
