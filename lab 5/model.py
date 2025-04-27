# model.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_layers=[], output_size=10, activation_fn=nn.ReLU):
        super(MLP, self).__init__()
        layers = []

        previous_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation_fn())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
