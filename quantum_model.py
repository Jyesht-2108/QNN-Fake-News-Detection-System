"""
Quantum Neural Network Model for Fake News Detection
====================================================
This module implements a variational quantum classifier using PennyLane.
It includes quantum feature encoding and a parameterized quantum circuit.
"""

import pennylane as qml
import numpy as np
from typing import Tuple, List
import pickle


class QuantumNeuralNetwork:
    """
    Variational Quantum Classifier for binary classification.
    
    This class implements a hybrid quantum-classical neural network with:
    - Amplitude encoding for classical features
    - Variational quantum circuit with trainable parameters
    - Measurement-based classification
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        device_name: str = 'default.qubit'
    ):
        """
        Initialize the quantum neural network.
        
        Args:
            n_qubits: Number of qubits (should match number of features)
            n_layers: Number of variational layers in the circuit
            device_name: PennyLane device to use ('default.qubit' for simulation)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name
        
        # Create quantum device
        self.dev = qml.device(device_name, wires=n_qubits)
        
        # Calculate number of parameters needed
        # Each layer has: n_qubits rotations (3 params each) + n_qubits entangling gates
        self.n_params = n_layers * n_qubits * 3
        
        # Initialize parameters randomly
        self.params = np.random.randn(n_layers, n_qubits, 3) * 0.1
        
        print(f"Quantum Neural Network initialized:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Layers: {n_layers}")
        print(f"  Parameters: {self.n_params}")
        print(f"  Device: {device_name}")
    
    def amplitude_encoding(self, features: np.ndarray):
        """
        Encode classical features into quantum state using amplitude encoding.
        
        This normalizes the feature vector and encodes it as amplitudes of a quantum state.
        For n features, we need log2(n) qubits (rounded up).
        
        Args:
            features: Classical feature vector
        """
        # Normalize features to create valid quantum state
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        # Pad features to match 2^n_qubits if needed
        required_size = 2 ** self.n_qubits
        if len(features) < required_size:
            features = np.pad(features, (0, required_size - len(features)))
        elif len(features) > required_size:
            features = features[:required_size]
        
        # Use PennyLane's amplitude embedding
        qml.AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)
    
    def angle_encoding(self, features: np.ndarray):
        """
        Alternative: Encode classical features using angle encoding.
        
        Each feature is encoded as a rotation angle on a qubit.
        This is simpler but may be less expressive than amplitude encoding.
        
        Args:
            features: Classical feature vector (length should match n_qubits)
        """
        for i, feature in enumerate(features[:self.n_qubits]):
            qml.RY(feature * np.pi, wires=i)
    
    def variational_layer(self, params: np.ndarray):
        """
        Apply one variational layer to the quantum circuit.
        
        Each layer consists of:
        1. Parameterized rotations on each qubit (RX, RY, RZ)
        2. Entangling gates (CNOT) between adjacent qubits
        
        Args:
            params: Parameters for this layer (shape: [n_qubits, 3])
        """
        # Parameterized rotations on each qubit
        for i in range(self.n_qubits):
            qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=i)
        
        # Entangling layer: CNOT gates in a ring topology
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def quantum_circuit(self, features: np.ndarray, params: np.ndarray) -> float:
        """
        Complete quantum circuit for classification.
        
        Args:
            features: Input features to encode
            params: Variational parameters
            
        Returns:
            Expectation value of measurement (used for classification)
        """
        # Encode classical data
        self.amplitude_encoding(features)
        
        # Apply variational layers
        for layer_params in params:
            self.variational_layer(layer_params)
        
        # Measure expectation value of first qubit in Z basis
        # This gives a value between -1 and 1
        return qml.expval(qml.PauliZ(0))
    
    def create_qnode(self):
        """
        Create a QNode (quantum node) that can be executed and differentiated.
        
        Returns:
            QNode function
        """
        return qml.QNode(self.quantum_circuit, self.dev, interface='autograd')
    
    def predict_proba(self, features: np.ndarray) -> float:
        """
        Predict probability for a single sample.
        
        Args:
            features: Feature vector for one sample
            
        Returns:
            Probability of class 1 (fake news)
        """
        qnode = self.create_qnode()
        expectation = qnode(features, self.params)
        
        # Convert expectation value [-1, 1] to probability [0, 1]
        # expectation = 1 -> prob = 0 (real news)
        # expectation = -1 -> prob = 1 (fake news)
        probability = (1 - expectation) / 2
        
        return float(probability)
    
    def predict(self, features: np.ndarray, threshold: float = 0.5) -> int:
        """
        Predict class label for a single sample.
        
        Args:
            features: Feature vector for one sample
            threshold: Classification threshold
            
        Returns:
            Predicted class (0 or 1)
        """
        prob = self.predict_proba(features)
        return 1 if prob >= threshold else 0
    
    def predict_batch(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for multiple samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            threshold: Classification threshold
            
        Returns:
            Array of predicted classes
        """
        predictions = []
        for features in X:
            predictions.append(self.predict(features, threshold))
        return np.array(predictions)
    
    def get_params(self) -> np.ndarray:
        """Get current model parameters."""
        return self.params.copy()
    
    def set_params(self, params: np.ndarray):
        """Set model parameters."""
        self.params = params.copy()
    
    def save(self, filepath: str):
        """Save model parameters to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'device_name': self.device_name
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model parameters from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.n_qubits = data['n_qubits']
            self.n_layers = data['n_layers']
            self.device_name = data['device_name']
            # Recreate device
            self.dev = qml.device(self.device_name, wires=self.n_qubits)
        print(f"Model loaded from {filepath}")
    
    def draw_circuit(self, sample_features: np.ndarray) -> str:
        """
        Draw the quantum circuit for visualization.
        
        Args:
            sample_features: Sample input features
            
        Returns:
            String representation of the circuit
        """
        qnode = self.create_qnode()
        return qml.draw(qnode)(sample_features, self.params)


def compute_cost(qnn: QuantumNeuralNetwork, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss for the quantum model.
    
    Args:
        qnn: Quantum neural network instance
        X: Feature matrix
        y: True labels
        
    Returns:
        Average loss over all samples
    """
    total_loss = 0.0
    epsilon = 1e-7  # For numerical stability
    
    for features, label in zip(X, y):
        pred_prob = qnn.predict_proba(features)
        # Binary cross-entropy
        loss = -(label * np.log(pred_prob + epsilon) + 
                (1 - label) * np.log(1 - pred_prob + epsilon))
        total_loss += loss
    
    return total_loss / len(X)


def compute_accuracy(qnn: QuantumNeuralNetwork, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        qnn: Quantum neural network instance
        X: Feature matrix
        y: True labels
        
    Returns:
        Accuracy score (0 to 1)
    """
    predictions = qnn.predict_batch(X)
    return np.mean(predictions == y)


if __name__ == "__main__":
    """
    Example usage and testing of the quantum model.
    """
    print("=" * 60)
    print("Testing Quantum Neural Network")
    print("=" * 60)
    
    # Create a simple QNN
    n_features = 4
    qnn = QuantumNeuralNetwork(n_qubits=n_features, n_layers=2)
    
    # Test with random data
    print("\nTesting with random data...")
    X_test = np.random.randn(5, n_features)
    y_test = np.random.randint(0, 2, 5)
    
    print(f"Input shape: {X_test.shape}")
    print(f"Labels: {y_test}")
    
    # Make predictions
    predictions = qnn.predict_batch(X_test)
    print(f"Predictions: {predictions}")
    
    # Compute metrics
    accuracy = compute_accuracy(qnn, X_test, y_test)
    loss = compute_cost(qnn, X_test, y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")
    
    # Draw circuit
    print("\nQuantum Circuit:")
    print(qnn.draw_circuit(X_test[0]))
    
    print("\n" + "=" * 60)
    print("Quantum model ready!")
    print("=" * 60)
