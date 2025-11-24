# Quick Start Guide 🚀

Get up and running with the Quantum Fake News Detector in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2-3 GB free disk space
- Internet connection (for downloading packages)

## Installation (3 steps)

### Step 1: Clone/Download the Project

```bash
cd quantum-fake-news-detection
```

### Step 2: Run Setup

```bash
python setup.py
```

This will:
- Check your Python version
- Create necessary directories
- Install all dependencies
- Download NLTK data
- Test the installation

**Note**: Installation takes 5-10 minutes depending on your internet speed.

### Step 3: Get a Dataset

**Option A - Quick Demo (Recommended for first try)**
```bash
python download_dataset.py
# Choose option 2: Create sample dataset
```

**Option B - Real Dataset**
```bash
python download_dataset.py
# Choose option 1: Download LIAR dataset
# OR manually download WELFake from https://mldata.vn/english/welfake
```

## Running the Project

### Quick Demo (No training needed)

```bash
python demo.py
```

This runs a quick demo with sample news articles.

### Full Training Pipeline

```bash
python train.py
```

**What happens:**
1. Loads and preprocesses the dataset
2. Creates quantum neural network
3. Trains for 50 epochs (~10-30 minutes)
4. Evaluates on test set
5. Saves model and generates plots

**Output files** (in `results/` directory):
- `quantum_model.pkl` - Trained model
- `preprocessor.pkl` - Text preprocessor
- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Test results
- `metrics.json` - Performance metrics

### Test Robustness

```bash
python robustness.py
```

Tests the model against adversarial text attacks.

## Configuration

### Fast Testing (Recommended for first run)

Edit `train.py` and set:
```python
SAMPLE_SIZE = 200      # Use only 200 samples
N_FEATURES = 4         # Fewer features
N_QUBITS = 4          # Fewer qubits
N_LAYERS = 2          # Simpler circuit
EPOCHS = 20           # Fewer epochs
```

Training time: ~5 minutes

### Full Training (Better accuracy)

Edit `train.py` and set:
```python
SAMPLE_SIZE = None     # Use all data
N_FEATURES = 8        # More features
N_QUBITS = 8         # More qubits
N_LAYERS = 3         # Deeper circuit
EPOCHS = 100         # More epochs
```

Training time: ~30-60 minutes

## Troubleshooting

### "No module named 'pennylane'"

```bash
pip install -r requirements.txt
```

### "Dataset not found"

The code will automatically use synthetic data. For real results:
```bash
python download_dataset.py
```

### Training is too slow

Reduce the configuration:
- Set `SAMPLE_SIZE = 200` in `train.py`
- Set `N_QUBITS = 4` and `N_LAYERS = 2`
- Set `EPOCHS = 20`

### Out of memory

Reduce `BATCH_SIZE` in `train.py`:
```python
BATCH_SIZE = 5  # Instead of 10
```

## Expected Results

With the sample dataset (200 samples, 20 epochs):
- **Training time**: ~5 minutes
- **Accuracy**: 60-75%
- **Loss**: Decreasing trend

With full WELFake dataset (1000+ samples, 50 epochs):
- **Training time**: ~30 minutes
- **Accuracy**: 75-85%
- **Loss**: Smooth convergence

## Next Steps

1. ✅ Run the quick demo
2. ✅ Train with sample data
3. ✅ Download real dataset
4. ✅ Train full model
5. ✅ Test robustness
6. 📖 Read full README.md
7. 🔧 Experiment with configurations

## Common Commands

```bash
# Setup
python setup.py

# Get dataset
python download_dataset.py

# Train model
python train.py

# Test robustness
python robustness.py

# Interactive demo
python demo.py

# Test individual modules
python data_preprocessing.py
python quantum_model.py
```

## File Overview

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `quantum_model.py` | Quantum circuit implementation |
| `data_preprocessing.py` | Text preprocessing |
| `robustness.py` | Adversarial testing |
| `demo.py` | Interactive demo |
| `config.py` | Configuration settings |
| `requirements.txt` | Dependencies |

## Getting Help

1. Check `README.md` for detailed documentation
2. Review inline code comments
3. Check error messages carefully
4. Ensure all dependencies are installed

## Performance Tips

**For faster training:**
- Use fewer samples (`SAMPLE_SIZE = 200`)
- Reduce qubits (`N_QUBITS = 4`)
- Fewer layers (`N_LAYERS = 2`)
- Fewer epochs (`EPOCHS = 20`)

**For better accuracy:**
- Use more data (`SAMPLE_SIZE = None`)
- More qubits (`N_QUBITS = 8`)
- Deeper circuit (`N_LAYERS = 4`)
- More epochs (`EPOCHS = 100`)

---

**Ready to start? Run:**
```bash
python setup.py
python demo.py
```

Happy quantum computing! 🔬✨
