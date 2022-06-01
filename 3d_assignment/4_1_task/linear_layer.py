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
y = linear_torch(x)

print(f"x: {x}")
print(f"output y: {y}\n")

# Loss from left outside
grad_y = torch.tensor([[1, 2], [2, 3]],
                      dtype=torch.float32)
grad_y_t = torch.tensor([grad_y[0].resize(2, 1).t(),
                       grad_y[1].resize(2, 1).t()],
                      dtype=torch.float32)


print(f"grad y: {grad_y}")
print(f"grad y_t: {grad_y_t}\n")

# Calculate gradient for x
grad_x = grad_y.matmul(w)

# Calculate gradient for x
grad_w = grad_y_t.matmul(x)

print(f"grad x: {grad_x}")
print(f"grad w: {grad_w}")


# Das woanders hin, sie in den aufgaben 4.2 das hat ne struktur
class Layer(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super(Layer, self).__init__()

        # Storage for weights from built in layer
        self.weights = torch.nn.Parameter(
            torch.Tensor(n_features_in, n_features_out))

    def forward(self, input):
        return torch.matmult(input, self.weights)
