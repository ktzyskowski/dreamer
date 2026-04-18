import torch.nn as nn


class Dreamer(nn.Module):
    def __init__(self, dream_horizon: int, n_dreams: int = -1):
        super().__init__()
        self.dream_horizon = dream_horizon
        self.n_dreams = n_dreams

    def dream(self):
        pass
