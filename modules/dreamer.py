from modules.models.actor import Actor
from modules.models.critic import Critic
from modules.models.world_model import WorldModel


class Dreamer:
    def __init__(self):
        self.world_model = WorldModel()
        self.actor = Actor()
        self.critic = Critic()

    def save_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"

    def load_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"


# import torch
# import torch.nn as nn


# # Sequence model
# # h_t = f(h_t-1, z_t-1, a_t-1)
# class SequenceModel(nn.Module):
#     def __init__(self, hidden_dim, latent_dim, action_dim):
#         super().__init__()


# # Encoder p2
# # z_t ~ p(z_t | h_t, e_t)
# # e_t = f(x_t)
# class PosteriorModel(nn.Module):
#     def __init__(self):
#         super().__init__()


# # Dynamics predictor
# # z_t ~ p(z_t | h_t)
# class PriorModel(nn.Module):
#     def __init__(self):
#         super().__init__()
