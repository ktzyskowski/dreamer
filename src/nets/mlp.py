from typing import Optional, Type

from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron class."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: Type[nn.Module],
        output_activation: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        # internal MLP layers
        layers = []

        # input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation())

        # hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(activation())

        # output layer + optional activation
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if output_activation:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y
