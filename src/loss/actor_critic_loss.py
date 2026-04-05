from torch import Tensor, nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from src.models.critic import DualCritic
from src.data import DreamOutput
from src.util.functions import calculate_lambda_returns
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

        self.moving_average = ExpMovingAverage(decay=config.actor_critic_loss.ema_decay)
        self.eta = config.actor_critic_loss.eta
        self.lamda = config.actor_critic_loss.lamda
        self.gamma = config.actor_critic_loss.gamma

        # see world_model_loss.py
        self.metrics = {}

    def forward(self, dream_output: DreamOutput, critic: DualCritic):
        # reset metrics accumulator
        self.metrics = {}

        # calculate lambda returns, required by both actor and critic loss functions
        critic_logits = critic.fast(dream_output["full_states"])
        critic_values = self.two_hot.decode(critic.slow(dream_output["full_states"]))
        continues = torch.distributions.Bernoulli(logits=dream_output["predicted_continue_logits"]).probs
        rewards = self.two_hot.decode(dream_output["predicted_reward_logits"])
        lambda_returns = calculate_lambda_returns(
            rewards=rewards,
            continues=continues.squeeze(-1),  # remove trailing size 1 dimension
            values=critic_values,
            gamma=self.gamma,
            lamda=self.lamda,
        )

        critic_loss = self.calculate_critic_loss(lambda_returns, critic_logits)
        actor_loss = self.calculate_actor_loss(lambda_returns, critic_values, dream_output)
        loss = critic_loss + actor_loss

        self.metrics["loss/actor"] = actor_loss.item()
        self.metrics["loss/critic"] = critic_loss.item()

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

        advantage = lambda_returns - critic_values
        summands = (advantage / torch.clamp(norm_term, min=1)).detach()
        summands = summands * action_log_probs
        loss = -summands.mean() - self.eta * entropy_term.mean()

        self.metrics["actor/entropy"] = entropy_term.mean().item()
        self.metrics["actor/advantage_mean"] = advantage.mean().item()
        self.metrics["actor/max_action_prob"] = action_probs.max(dim=-1).values.mean().item()
        self.metrics["returns/mean"] = lambda_returns.mean().item()
        self.metrics["returns/p5"] = p5.item()
        self.metrics["returns/p95"] = p95.item()
        self.metrics["returns/p95-p5"] = self.metrics["returns/p95"] - self.metrics["returns/p5"]
        self.metrics["returns/norm"] = norm_term.item()

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
