"""
Training Script for Hybrid Classical-Quantum Fake News Detector
================================================================
Implements strict paper specifications:
- Binary Cross Entropy loss
- Adam optimizer with lr=0.001
- Batch size=32
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

from quantum_model import HybridQuantumModel
from data_preprocessing import prepare_dataset

plt.switch_backend('Agg')

# ==========================================
# CONFIGURATION (Paper Specifications)
# ==========================================
DATASET_PATH = '/data/WELFake_Dataset.csv'
SAMPLE_SIZE = 5000  # Reduced for faster training on Colab
N_FEATURES = 8  # After PCA
N_QUBITS = 4  # Paper specification
N_LAYERS = 3  # Paper specification
BATCH_SIZE = 32  # Paper specification
LEARNING_RATE = 0.001  # Paper specification
EPOCHS = 20
RANDOM_STATE = 42

# Checkpoint path (will be inside results/)
CHECKPOINT_PATH = Path("results/checkpoint.pth")


class Trainer:
    """
    Trainer for hybrid quantum model.
    """
    
    def __init__(
        self,
        model: HybridQuantumModel,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Hybrid quantum model
            learning_rate: Learning rate for Adam optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Binary Cross Entropy loss (paper specification)
        self.criterion = nn.BCELoss()
        
        # Adam optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"Trainer initialized:")
        print(f"  Optimizer: Adam (lr={learning_rate})")
        print(f"  Loss: Binary Cross Entropy")
        print(f"  Device: {device}")
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device).float()
            # Ensure labels are float and have shape (batch_size, 1)
            batch_y = batch_y.to(self.device).float().unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            # 🔧 Ensure outputs and targets have the same dtype
            if outputs.dtype != batch_y.dtype:
                outputs = outputs.to(batch_y.dtype)
            
            # Compute loss
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader) -> tuple:
        """Evaluate model on validation set."""
        self.model.eval()
        
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Evaluating"):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float().unsqueeze(1)
                
                # Forward pass
                outputs = self.model(batch_x)
                # 🔧 Match dtypes again just in case
                if outputs.dtype != batch_y.dtype:
                    outputs = outputs.to(batch_y.dtype)
                
                # Compute loss
                loss = self.criterion(outputs, batch_y)
                
                # Track metrics
                epoch_loss += loss.item()
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = epoch_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20
    ):
        """Train the model."""
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_acc = 0.0
        best_model_state = None

        # 🔧 Resume from checkpoint if it exists
        start_epoch = 0
        if CHECKPOINT_PATH.exists():
            print(f"\n✅ Found checkpoint at {CHECKPOINT_PATH}, loading...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", -1) + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            print(f"Resuming from epoch {start_epoch} with best_val_acc={best_val_acc:.4f}")
        
        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model (in-memory)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict()
            
            # 🔧 Save checkpoint every epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"💾 Checkpoint saved at {CHECKPOINT_PATH} (epoch {epoch})")
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()


def evaluate_model(model, test_loader, device, save_dir):
    """Evaluate model on test set."""
    print("\nEvaluating on test set...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(device).float()
            outputs = model(batch_x)
            
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    # Flatten arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    print(f"\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics


def main():
    print("=" * 70)
    print("HYBRID CLASSICAL-QUANTUM FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n[STEP 1] Data Preprocessing (BERT + PCA)")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Sample size: {SAMPLE_SIZE}")
    
    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(
            dataset_path=DATASET_PATH,
            n_features=N_FEATURES,
            sample_size=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Save preprocessor
        preprocessor.save('results/preprocessor.pkl')
        
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Please update DATASET_PATH in train.py")
        return
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    # 🔧 Store labels as float for BCE
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 2: Model Initialization
    print("\n[STEP 2] Hybrid Quantum Model Initialization")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HybridQuantumModel(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    
    # Draw circuit
    print("\nQuantum Circuit:")
    print(model.draw_circuit(X_train[0]))
    
    # Step 3: Training
    print("\n[STEP 3] Training")
    
    trainer = Trainer(model, learning_rate=LEARNING_RATE, device=device)
    trainer.train(train_loader, test_loader, epochs=EPOCHS)
    
    # Plot training history
    trainer.plot_training_history(save_path='results/training_history.png')
    
    # Step 4: Evaluation
    print("\n[STEP 4] Final Evaluation")
    metrics = evaluate_model(model, test_loader, device, save_dir='results')
    
    # Step 5: Save Model
    print("\n[STEP 5] Saving Model")
    model.save('results/quantum_model.pth')
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: results/quantum_model.pth")
    print(f"Preprocessor saved to: results/preprocessor.pkl")
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
