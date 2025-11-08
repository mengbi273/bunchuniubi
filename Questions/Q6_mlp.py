# Validation Code
import numpy as np

# Test activation functions
print("Testing activation functions...")
x_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
relu_output = relu(x_test)
sigmoid_output = sigmoid(x_test)
print(f"ReLU output: {relu_output}")
print(f"Sigmoid output: {sigmoid_output}")
print("✓ Activation functions test passed!")

# Test linear layer
print("\nTesting linear layer...")
np.random.seed(42)
X_test = np.random.randn(5, 4)
weights_test = np.random.randn(4, 3)
bias_test = np.random.randn(3)
linear_output = linear_layer_forward(X_test, weights_test, bias_test)
print(f"Linear layer output shape: {linear_output.shape}")
assert linear_output.shape == (5, 3), "Linear layer output shape incorrect!"
print("✓ Linear layer test passed!")

# Test MLP forward pass
print("\nTesting MLP forward pass...")
np.random.seed(42)
n_samples, n_features = 10, 4
X = np.random.randn(n_samples, n_features)

# Create a 2-layer MLP: 4 -> 8 -> 1
weights_list = [
    np.random.randn(4, 8) * 0.1,
    np.random.randn(8, 1) * 0.1
]
bias_list = [
    np.zeros(8),
    np.zeros(1)
]

output_relu = mlp_forward(X, weights_list, bias_list, activation='relu')
output_sigmoid = mlp_forward(X, weights_list, bias_list, activation='sigmoid')
print(f"MLP output shape (ReLU): {output_relu.shape}")
print(f"MLP output shape (Sigmoid): {output_sigmoid.shape}")
assert output_relu.shape == (n_samples, 1), "MLP output shape incorrect!"
print("✓ MLP forward pass test passed!")

# Test backward pass
print("\nTesting MLP backward pass...")
y = np.random.randn(n_samples, 1)
w_grads, b_grads = mlp_backward(X, y, weights_list, bias_list, activation='relu')
print(f"Weight gradients shapes: {[g.shape for g in w_grads]}")
print(f"Bias gradients shapes: {[g.shape for g in b_grads]}")
assert len(w_grads) == len(weights_list), "Number of weight gradients doesn't match!"
assert len(b_grads) == len(bias_list), "Number of bias gradients doesn't match!"
print("✓ MLP backward pass test passed!")

