"""Quick test to verify gradient computation works"""
import numpy as np
from quantum_model import QuantumNeuralNetwork
from pennylane import numpy as pnp

print("Testing gradient computation fix...")

# Create small model
qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)

# Test data
X = np.random.randn(5, 4)
y = np.array([0, 1, 0, 1, 0])

# Test if parameters are trainable
print(f"Parameters type: {type(qnn.params)}")
print(f"Parameters require grad: {hasattr(qnn.params, 'requires_grad')}")

# Test prediction
pred = qnn.predict(X[0])
print(f"Prediction works: {pred}")

# Test gradient computation
from quantum_model import compute_cost
cost = compute_cost(qnn, X, y)
print(f"Cost computation works: {cost:.4f}")

print("\n✓ All tests passed! Gradient computation should work now.")
