import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.returns import calculate_lambda_returns
from src.transforms.ema import ExpMovingAverage
from src.transforms.twohot import SymlogTwoHot
from src.util.probability import policy_distribution


class ActorCriticLoss(nn.Module):
    def __init__(
        self,
        two_hot_low: float,
        two_hot_high: float,
        two_hot_n_bins: int,
        discount: float = 0.99,
        trace_decay: float = 0.95,
        entropy_coefficient: float = 5e-2,
        slow_regularization_weight: float = 1.0,
        advantage_norm_decay: float = 0.99,
    ):
        super().__init__()
        self.discount = discount
        self.trace_decay = trace_decay
        self.entropy_coefficient = entropy_coefficient
        self.slow_regularization_weight = slow_regularization_weight
        self.advantage_norm = ExpMovingAverage(decay=advantage_norm_decay)
        self.symlog_two_hot = SymlogTwoHot(low=two_hot_low, high=two_hot_high, n_bins=two_hot_n_bins)

    def forward(
        self,
        dream_output: dict,
        fast_critic_logits: torch.Tensor,
        slow_critic_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        slow_values = self.symlog_two_hot.decode(slow_critic_logits)
        rewards = self.symlog_two_hot.decode(dream_output["predicted_reward_logits"])
        continues = torch.sigmoid(dream_output["predicted_continue_logits"]).squeeze(-1)

        lambda_returns = calculate_lambda_returns(
            rewards=rewards,
            continues=continues,
            values=slow_values,
            discount=self.discount,
            trace_decay=self.trace_decay,
        )

        actor_loss, actor_metrics = self.actor_loss(lambda_returns, slow_values, dream_output)
        critic_loss, critic_metrics = self.critic_loss(lambda_returns, fast_critic_logits, slow_critic_logits)
        loss = actor_loss + critic_loss

        metrics = {
            **actor_metrics,
            **critic_metrics,
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
        }
        return loss, metrics

    def actor_loss(
        self,
        lambda_returns: torch.Tensor,
        values: torch.Tensor,
        dream_output: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_distribution = policy_distribution(dream_output["action_logits"])
        action_log_probs = action_distribution.log_prob(dream_output["actions"])
        entropy = action_distribution.entropy()

        # Normalize advantages by an EMA of the (p95 - p5) return spread.
        percentile_high = torch.quantile(lambda_returns.flatten(), q=0.95)
        percentile_low = torch.quantile(lambda_returns.flatten(), q=0.05)
        advantage_normalizer = self.advantage_norm(percentile_high - percentile_low)

        advantage = lambda_returns - values
        normalized_advantage = (advantage / torch.clamp(advantage_normalizer, min=1)).detach()
        reinforce_term = normalized_advantage * action_log_probs
        loss = -reinforce_term.mean() - self.entropy_coefficient * entropy.mean()

        metrics = {
            "actor/entropy": entropy.mean(),
            "actor/advantage_mean": advantage.mean(),
            "actor/max_action_prob": action_distribution.probs.max(dim=-1).values.mean(),
            "returns/mean": lambda_returns.mean(),
            "returns/percentile_low": percentile_low,
            "returns/percentile_high": percentile_high,
            "returns/spread": percentile_high - percentile_low,
            "returns/normalizer": advantage_normalizer,
        }
        return loss, metrics

    def critic_loss(
        self,
        lambda_returns: torch.Tensor,
        fast_critic_logits: torch.Tensor,
        slow_critic_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # SymlogTwoHot.encode applies symlog internally; pass raw values.
        return_target = self.symlog_two_hot.encode(lambda_returns.detach())
        # Regularize against the slow critic's own distribution directly, with
        # no lossy decode/re-encode through two-hot expectations.
        slow_probs = F.softmax(slow_critic_logits.detach(), dim=-1)

        # cross_entropy expects class dim in position 1: (B, T, C) -> (B, C, T)
        return_loss = F.cross_entropy(
            fast_critic_logits.permute(0, 2, 1),
            return_target.permute(0, 2, 1),
        )
        slow_regularization_loss = F.cross_entropy(
            fast_critic_logits.permute(0, 2, 1),
            slow_probs.permute(0, 2, 1),
        )

        loss = return_loss + self.slow_regularization_weight * slow_regularization_loss
        metrics = {
            "critic/return_loss": return_loss,
            "critic/slow_regularization_loss": slow_regularization_loss,
        }
        return loss, metrics
