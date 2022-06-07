from numpy import gradient, transpose
import torch

"""
    For Task 4.1
"""

# Input data
x = torch.tensor([[7, 8, 9], [10, 11, 12]],
                 dtype=torch.float32,
                 requires_grad=True)

# Standard linear layer
linear_torch = torch.nn.Linear(in_features=3,
                               out_features=2,
                               bias=False)

# Manually set the weights in the layer
w = torch.tensor([[1, 2, 3], [4, 5, 6]],
                 dtype=torch.float32, requires_grad=True)

linear_torch.weight = torch.nn.Parameter(w)

# Calculate ouput
y = linear_torch.forward(x)

print(f"x: {x}")
print(f"output y: {y}\n")

# Loss from left outside
grad_y = torch.tensor([[1, 2], [2, 3]],
                      dtype=torch.float32)
y.backward(grad_y)


print(f"grad y: {grad_y}")
# print(f"grad y_t: {grad_y_t}\n")

# Calculate gradient for x
grad_x = grad_y.matmul(w)

# Calculate gradient for x
grad_w = linear_torch.weight.grad

print(f"grad x: {grad_x}")
print(f"grad w: {grad_w}")