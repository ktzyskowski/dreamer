from torch import Tensor, nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from src.data import DreamOutput, WorldModelInput
from src.util.functions import calculate_lambda_returns, symexp, symlog
from src.util.two_hot import TwoHot
from src.util.ema import ExpMovingAverage


class ActorCriticLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.two_hot = TwoHot(
            low=config.two_hot.low,
            high=config.two_hot.high,
            n_bins=config.two_hot.n_bins,
        )

        self.moving_average = ExpMovingAverage(decay=0.99)
        self.eta = config.actor_critic_loss.eta

    def forward(self, dream_output: DreamOutput, critic: nn.Module):
        # calculate lambda returns, required by both actor and critic loss functions
        critic_logits = critic(dream_output["full_states"])
        critic_values = self.two_hot.decode(critic_logits)
        continues = torch.distributions.Bernoulli(logits=dream_output["predicted_continue_logits"]).probs
        rewards = self.two_hot.decode(dream_output["predicted_reward_logits"])
        lambda_returns = calculate_lambda_returns(
            rewards=rewards,
            continues=continues.squeeze(-1),  # remove trailing size 1 dimension
            values=critic_values,
            gamma=0.997,
            lamda=0.95,
        )

        critic_loss = self.calculate_critic_loss(lambda_returns, critic_logits)
        actor_loss = self.calculate_actor_loss(lambda_returns, critic_values, dream_output)
        loss = critic_loss + actor_loss
        return loss

    def calculate_actor_loss(
        self,
        lambda_returns: Tensor,
        critic_values: Tensor,
        dream_output: DreamOutput,
    ):
        action_probs = dream_output["action_probs"]
        action_indices = dream_output["actions"].argmax(-1)
        action_dist = Categorical(probs=action_probs)
        action_log_probs = action_dist.log_prob(action_indices)
        entropy_term = action_dist.entropy()

        # exponentially moving average over difference between 95th and 5th percentile
        p95 = torch.quantile(lambda_returns.flatten(), q=0.95)
        p5 = torch.quantile(lambda_returns.flatten(), q=0.05)
        norm_term = self.moving_average(p95 - p5)

        summands = ((lambda_returns - critic_values) / torch.clamp(norm_term, min=1)).detach()
        summands = summands * action_log_probs
        loss = -summands.mean() - self.eta * entropy_term.mean()
        return loss

    def calculate_critic_loss(
        self,
        lambda_returns: Tensor,
        critic_logits: Tensor,
    ):
        lambda_return_logits = self.two_hot.encode(lambda_returns.detach())
        # permute: (batch, sequence, classes) -> (batch, classes, sequence)
        # because cross_entropy expects class dimension in 2nd position
        critic_loss = F.cross_entropy(
            critic_logits.permute(0, 2, 1),
            lambda_return_logits.permute(0, 2, 1),
        )
        return critic_loss
