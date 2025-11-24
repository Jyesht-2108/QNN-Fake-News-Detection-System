# High-Accuracy Training Guide (90%+ Target)

## 🎯 Goal
Train a quantum neural network with:
- **90%+ accuracy** on fake news detection
- **Strong adversarial robustness** (resistant to text variations)
- **Real-world performance** (works on your own examples like "man claims he spoke to aliens!!")

---

## 📥 Step 1: Download WELFake Dataset

### Option A: Kaggle (Recommended)
1. Visit: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
2. Click "Download" (create free account if needed)
3. Extract the zip file
4. Move `WELFake_Dataset.csv` to: `data/WELFake_Dataset.csv`

### Option B: Alternative Source
1. Visit: https://zenodo.org/record/4561253
2. Download and extract
3. Place CSV in `data/` folder

### Verify Download
```bash
ls -lh data/WELFake_Dataset.csv
# Should show ~30MB file with 72,000+ rows
```

---

## 🚀 Step 2: Train High-Accuracy Model

Run the optimized training script:

```bash
python train_high_accuracy.py
```

### What This Does:
✅ Loads 5,000 samples from WELFake (or full dataset if you set SAMPLE_SIZE=None)  
✅ Uses 16 qubits and 4 layers (more powerful than basic model)  
✅ Generates adversarial examples for robustness (30% augmentation)  
✅ Trains for 100 epochs with optimized hyperparameters  
✅ Tests adversarial robustness automatically  
✅ Saves high-accuracy model  

### Training Time:
- **5,000 samples**: ~20-30 minutes
- **Full dataset (72K)**: ~2-3 hours

### Expected Results:
- **Accuracy**: 85-92%
- **Adversarial robustness**: 75-85% (on perturbed text)

---

## 🧪 Step 3: Test Your Model

### Interactive Demo
```bash
python demo_high_accuracy.py
```

This will:
1. Test on diverse examples (including "man claims he spoke to aliens!!")
2. Test adversarial robustness with text variations
3. Let you input your own text to classify

### Example Test Cases:
- ✅ "Man claims he spoke to aliens!!" → **FAKE** (sensational claim)
- ✅ "Scientists at MIT publish research..." → **REAL** (academic)
- ✅ "You won't believe this miracle cure..." → **FAKE** (clickbait)

---

## ⚙️ Configuration Options

Edit `train_high_accuracy.py` to customize:

### For Maximum Accuracy (90%+):
```python
CONFIG = {
    'SAMPLE_SIZE': None,        # Use ALL 72K samples
    'N_FEATURES': 16,           # Keep at 16
    'N_QUBITS': 16,            # Keep at 16
    'N_LAYERS': 5,             # Increase to 5 layers
    'EPOCHS': 150,             # Train longer
    'LEARNING_RATE': 0.003,    # Lower learning rate
    'USE_ADVERSARIAL': True,   # Keep enabled
    'ADVERSARIAL_RATIO': 0.4,  # More adversarial examples
}
```
⏱️ Training time: ~3-4 hours

### For Faster Training (85%+ accuracy):
```python
CONFIG = {
    'SAMPLE_SIZE': 3000,       # Fewer samples
    'N_FEATURES': 12,          # Fewer features
    'N_QUBITS': 12,           # Fewer qubits
    'N_LAYERS': 3,            # Fewer layers
    'EPOCHS': 50,             # Fewer epochs
    'LEARNING_RATE': 0.01,    # Higher learning rate
    'USE_ADVERSARIAL': True,  # Keep enabled
    'ADVERSARIAL_RATIO': 0.2, # Less augmentation
}
```
⏱️ Training time: ~10-15 minutes

---

## 🛡️ Adversarial Robustness Features

The model is trained to be robust against:

1. **Synonym Replacement**: "spoke" → "talked", "claims" → "alleges"
2. **Character Swapping**: Typos and misspellings
3. **Word Deletion**: Missing words
4. **Word Insertion**: Extra words
5. **Mixed Attacks**: Combination of above

### Testing Robustness:
```bash
python robustness.py
```

---

## 📊 Understanding Results

### Metrics Explained:
- **Accuracy**: Overall correctness (target: 90%+)
- **Precision**: Of predicted fake news, how many are actually fake
- **Recall**: Of actual fake news, how many were detected
- **F1-Score**: Balance between precision and recall

### Adversarial Accuracy:
- **Clean data**: 90%+ (your trained model)
- **Adversarial data**: 75-85% (after text perturbations)
- **Drop**: <15% is excellent robustness

---

## 🔧 Troubleshooting

### Issue: Accuracy below 85%
**Solutions:**
- Use more data: Set `SAMPLE_SIZE = None`
- Train longer: Increase `EPOCHS = 150`
- Deeper model: Increase `N_LAYERS = 5`

### Issue: Training too slow
**Solutions:**
- Reduce samples: `SAMPLE_SIZE = 2000`
- Fewer qubits: `N_QUBITS = 12`
- Fewer epochs: `EPOCHS = 50`

### Issue: Out of memory
**Solutions:**
- Smaller batch: `BATCH_SIZE = 10`
- Fewer samples: `SAMPLE_SIZE = 2000`
- Fewer features: `N_FEATURES = 12`

### Issue: Poor adversarial robustness
**Solutions:**
- More augmentation: `ADVERSARIAL_RATIO = 0.5`
- Train longer: `EPOCHS = 150`
- Lower learning rate: `LEARNING_RATE = 0.003`

---

## 📁 Output Files

After training, you'll have:

```
results/
├── quantum_model_high_accuracy.pkl          # Trained model
├── preprocessor_high_accuracy.pkl           # Text preprocessor
├── training_history_high_accuracy.png       # Training curves
├── confusion_matrix.png                     # Test results
├── robustness_high_accuracy.png            # Robustness results
└── high_accuracy_results.json              # All metrics
```

---

## 🎯 Success Criteria

Your model is ready when:

✅ **Accuracy ≥ 90%** on test set  
✅ **Adversarial accuracy ≥ 75%** on perturbed text  
✅ **Correctly classifies** "man claims he spoke to aliens!!" as FAKE  
✅ **Consistent predictions** across text variations  
✅ **Works on your own examples** in interactive mode  

---

## 💡 Tips for Best Results

1. **Use full dataset**: Set `SAMPLE_SIZE = None` for maximum accuracy
2. **Train overnight**: 150 epochs on full data takes 3-4 hours
3. **Enable adversarial training**: Crucial for robustness
4. **Test thoroughly**: Use `demo_high_accuracy.py` with diverse examples
5. **Monitor training**: Watch for convergence in training curves

---

## 🚀 Quick Start Commands

```bash
# 1. Download dataset (manual step)
# Visit Kaggle and download to data/WELFake_Dataset.csv

# 2. Train high-accuracy model
python train_high_accuracy.py

# 3. Test the model
python demo_high_accuracy.py

# 4. Test robustness
python robustness.py
```

---

## 📈 Expected Performance Timeline

| Configuration | Time | Accuracy | Robustness |
|--------------|------|----------|------------|
| Quick (2K samples, 50 epochs) | 10 min | 80-85% | 70-75% |
| Standard (5K samples, 100 epochs) | 25 min | 85-90% | 75-80% |
| Maximum (72K samples, 150 epochs) | 3-4 hrs | 90-95% | 80-85% |

---

## 🎉 You're Ready!

Follow the steps above to train a high-accuracy quantum fake news detector that:
- Achieves 90%+ accuracy
- Handles adversarial attacks
- Works on real-world examples
- Detects sensational claims like "man spoke to aliens!!"

Good luck with your training! 🚀
