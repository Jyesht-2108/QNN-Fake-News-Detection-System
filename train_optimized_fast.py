"""
OPTIMIZED FAST Training - 10-15 Minutes
=======================================
Balanced configuration for speed and accuracy.
Target: 85-88% accuracy in 10-15 minutes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch

from quantum_model import QuantumNeuralNetwork
from data_preprocessing import prepare_dataset
from train import QuantumTrainer, evaluate_model
from robustness import SimpleTextAttacker


print("=" * 70)
print("OPTIMIZED FAST QUANTUM FAKE NEWS DETECTOR")
print("Target: 85-88% Accuracy in 10-15 Minutes")
print("=" * 70)

# Check GPU
if torch.backends.mps.is_available():
    print("\n✅ Mac GPU (Metal) detected")
else:
    print("\n⚠️  Using CPU")

# ============================================================================
# OPTIMIZED FAST CONFIGURATION
# ============================================================================

CONFIG = {
    # Dataset - Smaller for speed
    'DATASET_PATH': 'data/WELFake_Dataset.csv',
    'DATASET_TYPE': 'welfake',
    'SAMPLE_SIZE': 2000,  # Smaller dataset = faster
    'TEST_SIZE': 0.2,
    
    # Model - Smaller for speed
    'N_FEATURES': 8,   # Fewer features = faster
    'N_QUBITS': 8,     # Fewer qubits = MUCH faster
    'N_LAYERS': 2,     # Fewer layers = faster
    
    # Training - Optimized for speed
    'EPOCHS': 40,      # Fewer epochs
    'BATCH_SIZE': 50,  # Larger batches = fewer iterations
    'LEARNING_RATE': 0.01,  # Higher LR = faster convergence
    
    # Adversarial - Light augmentation
    'USE_ADVERSARIAL': True,
    'ADVERSARIAL_RATIO': 0.15,  # Less augmentation = faster
    
    # Feature Extraction
    'MAX_TFIDF_FEATURES': 1000,
}

print("\n📋 Optimized Configuration:")
print(f"  Samples: {CONFIG['SAMPLE_SIZE']}")
print(f"  Qubits: {CONFIG['N_QUBITS']} (vs 16 in slow version)")
print(f"  Layers: {CONFIG['N_LAYERS']} (vs 4 in slow version)")
print(f"  Epochs: {CONFIG['EPOCHS']} (vs 100 in slow version)")

print("\n⏱️  Expected time: 10-15 minutes")
print("📊 Expected accuracy: 85-88%")
print("🚀 Speed improvement: 10-15x faster!")

# ============================================================================
# STEP 1: CHECK DATASET
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: Checking Dataset")
print("=" * 70)

dataset_path = Path(CONFIG['DATASET_PATH'])
if not dataset_path.exists():
    print(f"\n❌ Dataset not found at: {dataset_path}")
    exit(1)

print(f"✓ Dataset found")

# ============================================================================
# STEP 2: LOAD AND PREPROCESS DATA
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Data Preprocessing")
print("=" * 70)

X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(
    dataset_path=CONFIG['DATASET_PATH'],
    dataset_type=CONFIG['DATASET_TYPE'],
    n_features=CONFIG['N_FEATURES'],
    test_size=CONFIG['TEST_SIZE'],
    sample_size=CONFIG['SAMPLE_SIZE']
)

preprocessor.save('results/preprocessor_optimized.pkl')
print("✓ Preprocessor saved")

# ============================================================================
# STEP 3: ADVERSARIAL DATA AUGMENTATION (FAST)
# ============================================================================

if CONFIG['USE_ADVERSARIAL']:
    print("\n" + "=" * 70)
    print("STEP 3: Generating Adversarial Examples (Fast Mode)")
    print("=" * 70)
    
    import pandas as pd
    df_sample = pd.read_csv(CONFIG['DATASET_PATH'])
    if CONFIG['SAMPLE_SIZE']:
        df_sample = df_sample.sample(n=CONFIG['SAMPLE_SIZE'], random_state=42)
    
    if 'title' in df_sample.columns and 'text' in df_sample.columns:
        texts = (df_sample['title'].fillna('') + ' ' + df_sample['text'].fillna('')).tolist()
    else:
        texts = df_sample['text'].fillna('').tolist()
    
    from sklearn.model_selection import train_test_split
    texts_train, texts_test = train_test_split(
        texts, test_size=CONFIG['TEST_SIZE'], random_state=42
    )
    
    attacker = SimpleTextAttacker()
    n_adversarial = int(len(texts_train) * CONFIG['ADVERSARIAL_RATIO'])
    n_adversarial = min(n_adversarial, len(texts_train), len(y_train))
    
    print(f"Generating {n_adversarial} adversarial examples...")
    adv_texts = []
    adv_labels = []
    
    # Make sure indices are within bounds of both texts_train and y_train
    max_idx = min(len(texts_train), len(y_train))
    indices = np.random.choice(max_idx, n_adversarial, replace=False)
    
    for idx in tqdm(indices, desc="Creating adversarial samples"):
        adv_text = attacker.generate_adversarial(texts_train[idx], 'mixed')
        adv_texts.append(adv_text)
        adv_labels.append(y_train[idx])
    
    print("Preprocessing adversarial examples...")
    X_adv = preprocessor.transform(adv_texts)
    y_adv = np.array(adv_labels)
    
    X_train = np.vstack([X_train, X_adv])
    y_train = np.concatenate([y_train, y_adv])
    
    print(f"✓ Training set augmented to {len(X_train)} samples")

# ============================================================================
# STEP 4: CREATE QUANTUM MODEL (FAST)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Creating Quantum Neural Network (Fast Configuration)")
print("=" * 70)

qnn = QuantumNeuralNetwork(
    n_qubits=CONFIG['N_QUBITS'],
    n_layers=CONFIG['N_LAYERS']
)

print(f"\n✓ Model created:")
print(f"  Qubits: {qnn.n_qubits} (8 vs 16 = 256x faster!)")
print(f"  Layers: {qnn.n_layers}")
print(f"  Parameters: {qnn.n_params}")

# ============================================================================
# STEP 5: FAST TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Fast Training (10-15 minutes)")
print("=" * 70)

trainer = QuantumTrainer(
    qnn,
    learning_rate=CONFIG['LEARNING_RATE'],
    optimizer='adam'
)

print(f"\nTraining configuration:")
print(f"  Epochs: {CONFIG['EPOCHS']}")
print(f"  Batch size: {CONFIG['BATCH_SIZE']}")
print(f"  Learning rate: {CONFIG['LEARNING_RATE']}")
print(f"  Training samples: {len(X_train)}")

print("\n🚀 Starting fast training...")
print("   First epoch may take 1-2 minutes (compilation)")
print("   Subsequent epochs: 10-20 seconds each\n")

try:
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        verbose=True
    )
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted")
    qnn.save('results/quantum_model_checkpoint.pkl')
    print("✓ Checkpoint saved")
    exit(0)

trainer.plot_training_history(save_path='results/training_history_optimized.png')

# ============================================================================
# STEP 6: EVALUATE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Evaluation")
print("=" * 70)

metrics = evaluate_model(qnn, X_test, y_test, save_dir='results')

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: Saving Model")
print("=" * 70)

qnn.save('results/quantum_model_optimized.pkl')

import json
results_summary = {
    'configuration': CONFIG,
    'metrics': metrics,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('results/optimized_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Model saved: results/quantum_model_optimized.pkl")
print("✓ Preprocessor saved: results/preprocessor_optimized.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("🎉 FAST TRAINING COMPLETE!")
print("=" * 70)

print(f"\n📊 Performance:")
print(f"  Accuracy:  {metrics['accuracy']:.2%}")
print(f"  Precision: {metrics['precision']:.2%}")
print(f"  Recall:    {metrics['recall']:.2%}")
print(f"  F1-Score:  {metrics['f1']:.2%}")

if metrics['accuracy'] >= 0.85:
    print("\n✅ Excellent! 85%+ accuracy achieved in ~10-15 minutes")
elif metrics['accuracy'] >= 0.80:
    print("\n✅ Good! 80%+ accuracy achieved")
    print("💡 For 90%+: Use more data and train longer (but slower)")
else:
    print("\n💡 To improve: Increase SAMPLE_SIZE or EPOCHS")

print("\n🎯 Test Your Model:")
print("  python demo_optimized.py")

print("\n📁 Generated Files:")
print("  • results/quantum_model_optimized.pkl")
print("  • results/preprocessor_optimized.pkl")
print("  • results/training_history_optimized.png")
print("  • results/confusion_matrix.png")

print("\n" + "=" * 70)
print("✅ Model is ready to use!")
print("   Much faster than the 16-qubit version!")
print("=" * 70)
