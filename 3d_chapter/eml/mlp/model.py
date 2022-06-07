import torch
from torch import nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Flatten weil muss eine und nicht n Dimensionen für Input sein
        self.flatten = torch.nn.Flatten()
        self.model = torch.nn.Sequential(nn.Linear(28*28, 512),
                                         nn.ReLU(),
                                         nn.Linear(512, 512),
                                         nn.ReLU(),
                                         nn.Linear(512, 10))

    def forward(self, input):
        flattend_input = self.flatten(input)
        return self.model(flattend_input)
