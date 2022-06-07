import torch

# Das woanders hin, sie in den aufgaben 4.2 das hat ne struktur
class Layer(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super(Layer, self).__init__()

        # Storage for weights from built in layer
        self.weights = torch.nn.Parameter(
            torch.Tensor(n_features_in, n_features_out))

    def forward(self, input):
        return torch.matmult(input, self.weights)
