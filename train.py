"""
Training and Evaluation Script for Quantum Fake News Detector
=============================================================
This module handles training the quantum neural network and evaluating performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import pennylane as qml
from pennylane import numpy as pnp

from quantum_model import QuantumNeuralNetwork, compute_cost, compute_accuracy
from data_preprocessing import prepare_dataset


class QuantumTrainer:
    """
    Trainer class for quantum neural network with optimization and tracking.
    """
    
    def __init__(
        self,
        qnn: QuantumNeuralNetwork,
        learning_rate: float = 0.01,
        optimizer: str = 'adam'
    ):
        """
        Initialize the trainer.
        
        Args:
            qnn: Quantum neural network to train
            learning_rate: Learning rate for optimization
            optimizer: Optimizer type ('adam', 'sgd', 'adagrad')
        """
        self.qnn = qnn
        self.learning_rate = learning_rate
        
        # Initialize PennyLane optimizer
        if optimizer.lower() == 'adam':
            self.opt = qml.AdamOptimizer(stepsize=learning_rate)
        elif optimizer.lower() == 'sgd':
            self.opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
        elif optimizer.lower() == 'adagrad':
            self.opt = qml.AdagradOptimizer(stepsize=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"Trainer initialized with {optimizer} optimizer (lr={learning_rate})")
    
    def compute_gradient_cost(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Cost function for gradient computation.
        
        Args:
            params: Current parameters
            X: Training features
            y: Training labels
            
        Returns:
            Average loss
        """
        # Temporarily set parameters
        old_params = self.qnn.get_params()
        self.qnn.set_params(params)
        
        # Compute cost
        cost = compute_cost(self.qnn, X, y)
        
        # Restore parameters
        self.qnn.set_params(old_params)
        
        return cost
    
    def train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 10
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training labels
            batch_size: Number of samples per batch
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)
        
        epoch_losses = []
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Define cost function for this batch
            def cost_fn(params):
                self.qnn.set_params(params)
                return compute_cost(self.qnn, X_batch, y_batch)
            
            # Perform optimization step
            from pennylane import numpy as pnp
            params = pnp.array(self.qnn.get_params(), requires_grad=True)
            params, cost = self.opt.step_and_cost(cost_fn, params)
            self.qnn.set_params(params)
            
            epoch_losses.append(cost)
        
        # Compute metrics on full training set
        avg_loss = np.mean(epoch_losses)
        accuracy = compute_accuracy(self.qnn, X_train, y_train)
        
        return avg_loss, accuracy
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        loss = compute_cost(self.qnn, X, y)
        accuracy = compute_accuracy(self.qnn, X, y)
        return loss, accuracy
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 10,
        verbose: bool = True
    ):
        """
        Train the quantum neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print progress
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        best_val_acc = 0.0
        best_params = None
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            
            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = self.qnn.get_params()
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Restore best parameters
        if best_params is not None:
            self.qnn.set_params(best_params)
            print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training and validation metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
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
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


def evaluate_model(
    qnn: QuantumNeuralNetwork,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str = 'results'
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the trained model.
    
    Args:
        qnn: Trained quantum neural network
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Evaluating Model on Test Set")
    print("=" * 60)
    
    # Make predictions
    print("Making predictions...")
    y_pred = qnn.predict_batch(X_test)
    y_proba = np.array([qnn.predict_proba(x) for x in X_test])
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Print metrics
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Quantum Fake News Detector', fontsize=14, fontweight='bold')
    
    # Save confusion matrix
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cm_path = Path(save_dir) / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {cm_path}")
    plt.show()
    
    return metrics


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("QUANTUM FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Configuration
    DATASET_PATH = 'data/Enhanced_FakeNews_Dataset.csv'  # Update this path
    DATASET_TYPE = 'welfake'  # or 'liar'
    N_FEATURES = 8  # Number of quantum features
    N_QUBITS = 8  # Should match n_features
    N_LAYERS = 3  # Number of variational layers
    EPOCHS = 50
    BATCH_SIZE = 10
    LEARNING_RATE = 0.01
    SAMPLE_SIZE = None  # Use subset for faster training (set to None for full dataset)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n" + "=" * 60)
    print("STEP 1: Data Preprocessing")
    print("=" * 60)
    
    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(
            dataset_path=DATASET_PATH,
            dataset_type=DATASET_TYPE,
            n_features=N_FEATURES,
            sample_size=SAMPLE_SIZE
        )
        
        # Save preprocessor
        preprocessor.save('results/preprocessor.pkl')
        
    except FileNotFoundError:
        print(f"\nDataset not found at {DATASET_PATH}")
        print("Using synthetic data for demonstration...")
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 200
        X_train = np.random.randn(int(n_samples * 0.8), N_FEATURES)
        X_test = np.random.randn(int(n_samples * 0.2), N_FEATURES)
        y_train = np.random.randint(0, 2, int(n_samples * 0.8))
        y_test = np.random.randint(0, 2, int(n_samples * 0.2))
        
        print(f"Generated synthetic dataset:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Create a dummy preprocessor for synthetic data
        # This allows the demo to work with the trained model
        print("\nCreating preprocessor for synthetic data...")
        from data_preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor(n_features=N_FEATURES)
        
        # Fit on some dummy text data so it can be used later
        dummy_texts = [
            "This is a sample real news article about science and technology.",
            "FAKE NEWS ALERT! You won't believe this shocking revelation!",
            "Government officials announce new policy regarding infrastructure.",
            "Amazing miracle cure discovered! Doctors hate this one trick!",
        ]
        dummy_labels = np.array([0, 1, 0, 1])
        preprocessor.fit_transform(dummy_texts, dummy_labels)
        
        # Save the preprocessor
        preprocessor.save('results/preprocessor.pkl')
        print("✓ Preprocessor saved")
    
    # Step 2: Create quantum model
    print("\n" + "=" * 60)
    print("STEP 2: Quantum Model Initialization")
    print("=" * 60)
    
    qnn = QuantumNeuralNetwork(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS
    )
    
    # Display sample circuit
    print("\nSample Quantum Circuit:")
    print(qnn.draw_circuit(X_train[0]))
    
    # Step 3: Train the model
    print("\n" + "=" * 60)
    print("STEP 3: Training")
    print("=" * 60)
    
    trainer = QuantumTrainer(qnn, learning_rate=LEARNING_RATE, optimizer='adam')
    
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    # Plot training history
    trainer.plot_training_history(save_path='results/training_history.png')
    
    # Step 4: Evaluate the model
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)
    
    metrics = evaluate_model(qnn, X_test, y_test, save_dir='results')
    
    # Step 5: Save the trained model
    print("\n" + "=" * 60)
    print("STEP 5: Saving Model")
    print("=" * 60)
    
    qnn.save('results/quantum_model.pkl')
    
    # Save metrics
    import json
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to results/metrics.json")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved in '{results_dir}' directory:")
    print("  - quantum_model.pkl (trained model)")
    print("  - preprocessor.pkl (text preprocessor)")
    print("  - training_history.png (loss and accuracy curves)")
    print("  - confusion_matrix.png (test set confusion matrix)")
    print("  - metrics.json (evaluation metrics)")


if __name__ == "__main__":
    main()
