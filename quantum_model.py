"""
Quantum Neural Network Model for Fake News Detection
====================================================
Implements strict paper specifications:
- 4 qubits
- Angle encoding with Ry(pi * x_i)
- 3 variational layers with Rx, Ry, Rz + CNOT entanglement
- Pauli-Z measurement on qubit 0
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import pickle


class QuantumCircuit:
    """
    Quantum circuit following paper specifications.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        """
        Initialize quantum circuit.
        
        Args:
            n_qubits: Number of qubits (default: 4)
            n_layers: Number of variational layers (default: 3)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Calculate number of parameters: n_layers * n_qubits * 3 (Rx, Ry, Rz per qubit)
        self.n_params = n_layers * n_qubits * 3
        
        print(f"Quantum Circuit initialized:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Layers: {n_layers}")
        print(f"  Parameters: {self.n_params}")
    
    def angle_encoding(self, features):
        """
        Angle encoding: Ry(pi * x_i) for each feature.
        Maps 8 features to 4 qubits (2 features per qubit with re-uploading).
        
        Args:
            features: Input features (8-dimensional)
        """
        # Map 8 features to 4 qubits (2 features per qubit)
        for i in range(self.n_qubits):
            if i * 2 < len(features):
                qml.RY(np.pi * features[i * 2], wires=i)
            if i * 2 + 1 < len(features):
                qml.RY(np.pi * features[i * 2 + 1], wires=i)
    
    def variational_layer(self, params):
        """
        One variational layer with:
        1. Parameterized Rx, Ry, Rz on each qubit
        2. CNOT gates between adjacent qubits
        
        Args:
            params: Parameters for this layer (shape: [n_qubits, 3])
        """
        # Parameterized rotations
        for i in range(self.n_qubits):
            qml.RX(params[i, 0], wires=i)
            qml.RY(params[i, 1], wires=i)
            qml.RZ(params[i, 2], wires=i)
        
        # Entanglement: CNOT between adjacent qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def circuit(self, features, params):
        """
        Complete quantum circuit.
        
        Args:
            features: Input features (8-dimensional)
            params: Variational parameters (shape: [n_layers, n_qubits, 3])
            
        Returns:
            Expectation value of Pauli-Z on qubit 0
        """
        # Angle encoding
        self.angle_encoding(features)
        
        # Variational layers
        params_reshaped = params.reshape(self.n_layers, self.n_qubits, 3)
        for layer_params in params_reshaped:
            self.variational_layer(layer_params)
        
        # Measurement: Pauli-Z on qubit 0
        return qml.expval(qml.PauliZ(0))
    
    def create_qnode(self):
        """Create QNode for execution."""
        return qml.QNode(self.circuit, self.dev, interface='torch')


class HybridQuantumModel(nn.Module):
    """
    Hybrid Classical-Quantum Model for Fake News Detection.
    
    Architecture:
    Input (8-dim) -> Quantum Circuit -> Sigmoid -> Output Probability
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        """
        Initialize hybrid model.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize quantum circuit
        self.qcircuit = QuantumCircuit(n_qubits, n_layers)
        self.qnode = self.qcircuit.create_qnode()
        
        # Trainable quantum parameters
        self.q_params = nn.Parameter(
            torch.randn(self.qcircuit.n_params) * 0.1
        )
        
        print(f"Hybrid Quantum Model initialized with {self.qcircuit.n_params} trainable parameters")
    
    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input features (batch_size, 8)
            
        Returns:
            Output probabilities (batch_size, 1)
        """
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Get quantum circuit output (expectation value in [-1, 1])
            qout = self.qnode(x[i], self.q_params)
            outputs.append(qout)
        
        # Stack outputs
        outputs = torch.stack(outputs)
        
        # Convert expectation value [-1, 1] to probability [0, 1]
        # expectation = 1 -> prob = 0, expectation = -1 -> prob = 1
        probs = (1 - outputs) / 2
        
        # Apply sigmoid for additional non-linearity
        probs = torch.sigmoid(probs)
        
        return probs.unsqueeze(1)
    
    def save(self, filepath: str):
        """Save model parameters."""
        torch.save({
            'q_params': self.q_params.data,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.q_params.data = checkpoint['q_params']
        self.n_qubits = checkpoint['n_qubits']
        self.n_layers = checkpoint['n_layers']
        print(f"Model loaded from {filepath}")
    
    def draw_circuit(self, sample_input):
        """Draw the quantum circuit for visualization."""
        return qml.draw(self.qnode)(sample_input, self.q_params.detach())


if __name__ == "__main__":
    print("Testing Hybrid Quantum Model...")
    
    # Create model
    model = HybridQuantumModel(n_qubits=4, n_layers=3)
    
    # Test with random input
    x = torch.randn(2, 8)  # Batch of 2 samples, 8 features each
    
    print(f"\nInput shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().detach().numpy()}")
    
    # Draw circuit
    print("\nQuantum Circuit:")
    print(model.draw_circuit(x[0].detach().numpy()))
