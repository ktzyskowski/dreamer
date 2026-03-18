from torch import nn
import torch

from util.functions import calculate_lambda_returns


class ActorCriticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        # calculate lambda returns, required by both actor and critic loss functions
        continues = torch.distributions.Bernoulli(
            logits=batch["predicted_continue_logits"]
        ).probs
        lambda_returns = calculate_lambda_returns(
            rewards=batch["rewards"],
            continues=batch["continues"],
            values=batch["critic_values"],
            gamma=0.997,
            lamda=0.95,
        )

