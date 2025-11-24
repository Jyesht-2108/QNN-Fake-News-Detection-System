"""
High-Accuracy Training with Adversarial Robustness
==================================================
Optimized training for 90%+ accuracy with strong adversarial robustness.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from quantum_model import QuantumNeuralNetwork
from data_preprocessing import prepare_dataset
from train import QuantumTrainer, evaluate_model
from robustness import SimpleTextAttacker


print("=" * 70)
print("HIGH-ACCURACY QUANTUM FAKE NEWS DETECTOR")
print("Target: 90%+ Accuracy with Adversarial Robustness")
print("=" * 70)

# Check for GPU acceleration on Mac
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\n✓ GPU Acceleration: Metal Performance Shaders (MPS) available")
    print("  Using Mac GPU for faster training!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("\n✓ GPU Acceleration: CUDA available")
else:
    device = torch.device("cpu")
    print("\n⚠️  GPU Acceleration: Not available, using CPU")
    print("  Training will be slower but still work")

# ============================================================================
# OPTIMIZED CONFIGURATION FOR HIGH ACCURACY
# ============================================================================

CONFIG = {
    # Dataset
    'DATASET_PATH': 'data/WELFake_Dataset.csv',
    'DATASET_TYPE': 'welfake',
    'SAMPLE_SIZE': 5000,  # Use 5000 samples (increase to None for full dataset)
    'TEST_SIZE': 0.2,
    
    # Model Architecture (Optimized)
    'N_FEATURES': 16,  # More features = better representation
    'N_QUBITS': 16,    # More qubits = more expressiveness
    'N_LAYERS': 4,     # Deeper circuit = better learning
    
    # Training (Optimized)
    'EPOCHS': 100,     # More epochs for convergence
    'BATCH_SIZE': 32,  # Larger batches for GPU efficiency
    'LEARNING_RATE': 0.005,  # Lower LR for better convergence
    
    # Adversarial Training
    'USE_ADVERSARIAL': True,  # Enable adversarial training
    'ADVERSARIAL_RATIO': 0.3,  # 30% adversarial examples
    
    # Feature Extraction
    'MAX_TFIDF_FEATURES': 2000,  # More features before PCA
}

print("\n📋 Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# STEP 1: CHECK DATASET
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: Checking Dataset")
print("=" * 70)

dataset_path = Path(CONFIG['DATASET_PATH'])
if not dataset_path.exists():
    print(f"\n❌ Dataset not found at: {dataset_path}")
    print("\n📥 Please download WELFake dataset:")
    print("  1. Visit: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification")
    print("  2. Download and extract")
    print(f"  3. Place CSV at: {dataset_path.absolute()}")
    print("\nOr run: python download_welfake.py")
    exit(1)

print(f"✓ Dataset found: {dataset_path}")

# Check dataset size
import pandas as pd
df = pd.read_csv(dataset_path)
print(f"✓ Total samples in dataset: {len(df)}")
print(f"✓ Will use: {CONFIG['SAMPLE_SIZE'] or 'all'} samples")

# ============================================================================
# STEP 2: LOAD AND PREPROCESS DATA
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Data Preprocessing (This may take a few minutes...)")
print("=" * 70)

try:
    X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(
        dataset_path=CONFIG['DATASET_PATH'],
        dataset_type=CONFIG['DATASET_TYPE'],
        n_features=CONFIG['N_FEATURES'],
        test_size=CONFIG['TEST_SIZE'],
        sample_size=CONFIG['SAMPLE_SIZE']
    )
    
    # Save preprocessor
    preprocessor.save('results/preprocessor_high_accuracy.pkl')
    print("✓ Preprocessor saved")
    
except Exception as e:
    print(f"\n❌ Error during preprocessing: {e}")
    print("\nTrying with smaller sample size...")
    CONFIG['SAMPLE_SIZE'] = 2000
    
    X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(
        dataset_path=CONFIG['DATASET_PATH'],
        dataset_type=CONFIG['DATASET_TYPE'],
        n_features=CONFIG['N_FEATURES'],
        test_size=CONFIG['TEST_SIZE'],
        sample_size=CONFIG['SAMPLE_SIZE']
    )
    preprocessor.save('results/preprocessor_high_accuracy.pkl')

# ============================================================================
# STEP 3: ADVERSARIAL DATA AUGMENTATION
# ============================================================================

if CONFIG['USE_ADVERSARIAL']:
    print("\n" + "=" * 70)
    print("STEP 3: Generating Adversarial Examples for Robustness")
    print("=" * 70)
    
    # We need the original texts for adversarial generation
    # Load them again
    df_sample = pd.read_csv(CONFIG['DATASET_PATH'])
    if CONFIG['SAMPLE_SIZE']:
        df_sample = df_sample.sample(n=CONFIG['SAMPLE_SIZE'], random_state=42)
    
    # Get texts
    if 'title' in df_sample.columns and 'text' in df_sample.columns:
        texts = (df_sample['title'].fillna('') + ' ' + df_sample['text'].fillna('')).tolist()
    else:
        texts = df_sample['text'].fillna('').tolist()
    
    # Split texts same way as data
    from sklearn.model_selection import train_test_split
    texts_train, texts_test = train_test_split(
        texts, test_size=CONFIG['TEST_SIZE'], random_state=42
    )
    
    # Generate adversarial examples
    attacker = SimpleTextAttacker()
    n_adversarial = int(len(texts_train) * CONFIG['ADVERSARIAL_RATIO'])
    
    print(f"Generating {n_adversarial} adversarial examples...")
    adv_texts = []
    adv_labels = []
    
    # Make sure we have the right number of samples
    n_adversarial = min(n_adversarial, len(texts_train))
    indices = np.random.choice(len(texts_train), n_adversarial, replace=False)
    
    for i, idx in enumerate(tqdm(indices, desc="Creating adversarial samples")):
        original_text = texts_train[idx]
        original_label = y_train[idx]  # Get label from y_train at same index
        
        # Use mixed attacks for robustness
        adv_text = attacker.generate_adversarial(original_text, 'mixed')
        adv_texts.append(adv_text)
        adv_labels.append(original_label)
    
    # Preprocess adversarial examples
    print("Preprocessing adversarial examples...")
    X_adv = preprocessor.transform(adv_texts)
    y_adv = np.array(adv_labels)
    
    # Combine with original training data
    X_train_combined = np.vstack([X_train, X_adv])
    y_train_combined = np.concatenate([y_train, y_adv])
    
    print(f"✓ Training set augmented: {len(X_train)} → {len(X_train_combined)} samples")
    
    X_train = X_train_combined
    y_train = y_train_combined

# ============================================================================
# STEP 4: CREATE OPTIMIZED QUANTUM MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Creating Optimized Quantum Neural Network")
print("=" * 70)

# Use lightning.gpu device if available for faster simulation
try:
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        # Try to use GPU-accelerated device
        device_name = 'lightning.gpu'
        print(f"Attempting to use GPU-accelerated quantum device...")
    else:
        device_name = 'default.qubit'
except:
    device_name = 'default.qubit'

try:
    qnn = QuantumNeuralNetwork(
        n_qubits=CONFIG['N_QUBITS'],
        n_layers=CONFIG['N_LAYERS'],
        device_name=device_name
    )
    print(f"✓ Using device: {device_name}")
except:
    # Fallback to default device
    print(f"⚠️  GPU device not available, falling back to CPU")
    qnn = QuantumNeuralNetwork(
        n_qubits=CONFIG['N_QUBITS'],
        n_layers=CONFIG['N_LAYERS'],
        device_name='default.qubit'
    )

print(f"\n✓ Model created:")
print(f"  Qubits: {qnn.n_qubits}")
print(f"  Layers: {qnn.n_layers}")
print(f"  Parameters: {qnn.n_params}")
print(f"  Expressiveness: {'High' if qnn.n_layers >= 4 else 'Medium'}")

# ============================================================================
# STEP 5: TRAIN WITH OPTIMIZED HYPERPARAMETERS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Training (This will take 15-30 minutes...)")
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
print(f"  Validation samples: {len(X_test)}")

# Train with checkpointing
print("\n💡 Tip: Training will save checkpoints every 25 epochs")
print("   You can stop and resume training if needed\n")

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
    print("\n\n⚠️  Training interrupted by user")
    print("Saving current model state...")
    qnn.save('results/quantum_model_checkpoint.pkl')
    print("✓ Checkpoint saved. You can resume training later.")
    exit(0)

# Save training history
trainer.plot_training_history(save_path='results/training_history_high_accuracy.png')

# ============================================================================
# STEP 6: EVALUATE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Comprehensive Evaluation")
print("=" * 70)

metrics = evaluate_model(
    qnn,
    X_test,
    y_test,
    save_dir='results'
)

# ============================================================================
# STEP 7: TEST ADVERSARIAL ROBUSTNESS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: Testing Adversarial Robustness")
print("=" * 70)

from robustness import RobustnessTester

# Test on a subset for speed
test_size = min(100, len(texts_test))
test_texts_subset = texts_test[:test_size]
test_labels_subset = y_test[:test_size]

tester = RobustnessTester(qnn, preprocessor)
robustness_results = tester.test_robustness(
    test_texts_subset,
    test_labels_subset,
    attack_types=['synonym', 'char_swap', 'deletion', 'mixed']
)

# Plot robustness
tester.plot_robustness_results(
    robustness_results,
    save_path='results/robustness_high_accuracy.png'
)

# ============================================================================
# STEP 8: SAVE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 8: Saving High-Accuracy Model")
print("=" * 70)

qnn.save('results/quantum_model_high_accuracy.pkl')

# Save configuration and results
import json
results_summary = {
    'configuration': CONFIG,
    'metrics': metrics,
    'robustness': {k: v for k, v in robustness_results.items() if 'accuracy' in v},
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('results/high_accuracy_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Model saved: results/quantum_model_high_accuracy.pkl")
print("✓ Preprocessor saved: results/preprocessor_high_accuracy.pkl")
print("✓ Results saved: results/high_accuracy_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("🎉 HIGH-ACCURACY TRAINING COMPLETE!")
print("=" * 70)

print(f"\n📊 Final Performance:")
print(f"  Accuracy:  {metrics['accuracy']:.2%}")
print(f"  Precision: {metrics['precision']:.2%}")
print(f"  Recall:    {metrics['recall']:.2%}")
print(f"  F1-Score:  {metrics['f1']:.2%}")

if metrics['accuracy'] >= 0.90:
    print("\n✅ TARGET ACHIEVED: 90%+ Accuracy!")
elif metrics['accuracy'] >= 0.85:
    print("\n⚠️  Close to target (85%+). Try:")
    print("  • Increase SAMPLE_SIZE to use more data")
    print("  • Increase EPOCHS to 150")
    print("  • Increase N_LAYERS to 5")
else:
    print("\n💡 To improve accuracy:")
    print("  • Use full dataset (SAMPLE_SIZE = None)")
    print("  • Train longer (EPOCHS = 150)")
    print("  • Increase model size (N_QUBITS = 16, N_LAYERS = 5)")

print(f"\n🛡️  Adversarial Robustness:")
for attack_type, result in robustness_results.items():
    if 'accuracy' in result:
        print(f"  {attack_type}: {result['accuracy']:.2%}")

print("\n📁 Generated Files:")
print("  • results/quantum_model_high_accuracy.pkl")
print("  • results/preprocessor_high_accuracy.pkl")
print("  • results/training_history_high_accuracy.png")
print("  • results/confusion_matrix.png")
print("  • results/robustness_high_accuracy.png")
print("  • results/high_accuracy_results.json")

print("\n🎯 Test Your Model:")
print("  python demo_high_accuracy.py")

print("\n" + "=" * 70)
