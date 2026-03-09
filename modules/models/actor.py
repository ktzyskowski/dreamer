from torch import nn


class Actor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_state, no_grad=False):
        pass
