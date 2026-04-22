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
        beta_critic_dream: float = 1.0,
        beta_critic_real: float = 0.3,
    ):
        super().__init__()
        self.discount = discount
        self.trace_decay = trace_decay
        self.entropy_coefficient = entropy_coefficient
        self.slow_regularization_weight = slow_regularization_weight
        self.beta_critic_dream = beta_critic_dream
        self.beta_critic_real = beta_critic_real
        self.advantage_norm = ExpMovingAverage(decay=advantage_norm_decay)
        self.symlog_two_hot = SymlogTwoHot(low=two_hot_low, high=two_hot_high, n_bins=two_hot_n_bins)

    def forward(
        self,
        dream_output: dict,
        fast_critic_logits: torch.Tensor,
        slow_critic_logits: torch.Tensor,
        real_rewards: torch.Tensor,
        real_continues: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute actor + critic losses over dream rollouts and real replay states.

        Args:
            dream_output: output dict from Dreamer.dream(), with full_states (B, T, H+1, F),
                predicted_reward_logits (B, T, H, bins), predicted_continue_logits (B, T, H, 1).
            fast_critic_logits: fast critic evaluated on dream full_states, (B, T, H+1, bins).
            slow_critic_logits: slow critic evaluated on dream full_states, (B, T, H+1, bins).
            real_rewards: observed rewards from replay batch, (B, T).
            real_continues: observed continue flags (1 - done) from replay batch, (B, T).

        Returns:
            Scalar loss and metrics dict.
        """
        slow_values = self.symlog_two_hot.decode(slow_critic_logits)                      # (B, T, H+1)
        rewards = self.symlog_two_hot.decode(dream_output["predicted_reward_logits"])      # (B, T, H)
        continues = torch.sigmoid(dream_output["predicted_continue_logits"]).squeeze(-1)  # (B, T, H)

        lambda_returns = calculate_lambda_returns(
            rewards=rewards,
            continues=continues,
            values=slow_values,
            discount=self.discount,
            trace_decay=self.trace_decay,
        )  # (B, T, H+1)

        # index 0 = seed (real state from replay buffer); indices 1..H = imagined states
        dream_returns = lambda_returns[:, :, 1:]       # (B, T, H)
        dream_fast = fast_critic_logits[:, :, 1:, :]  # (B, T, H, bins)
        dream_slow = slow_critic_logits[:, :, 1:, :]  # (B, T, H, bins)

        # λ-returns over the real trajectory using observed rewards/continues.
        # rewards[:, :-1] drops the last reward since its successor state is outside the batch window.
        real_slow_values = self.symlog_two_hot.decode(slow_critic_logits[:, :, 0, :])  # (B, T)
        real_returns = calculate_lambda_returns(
            rewards=real_rewards[:, :-1],
            continues=real_continues[:, :-1],
            values=real_slow_values,
            discount=self.discount,
            trace_decay=self.trace_decay,
        )  # (B, T)
        real_fast = fast_critic_logits[:, :, 0, :]    # (B, T, bins)
        real_slow = slow_critic_logits[:, :, 0, :]    # (B, T, bins)

        actor_loss, actor_metrics = self.actor_loss(lambda_returns, slow_values, dream_output)

        dream_critic_loss, dream_return_loss, dream_slow_loss = self.critic_regression(
            dream_fast, dream_slow, dream_returns
        )
        real_critic_loss, real_return_loss, real_slow_loss = self.critic_regression(
            real_fast, real_slow, real_returns
        )

        critic_loss = self.beta_critic_dream * dream_critic_loss + self.beta_critic_real * real_critic_loss
        loss = actor_loss + critic_loss

        metrics = {
            **actor_metrics,
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
            "loss/critic/dream": dream_critic_loss,
            "loss/critic/real": real_critic_loss,
            "critic/dream/return_loss": dream_return_loss,
            "critic/dream/slow_reg_loss": dream_slow_loss,
            "critic/real/return_loss": real_return_loss,
            "critic/real/slow_reg_loss": real_slow_loss,
        }
        return loss, metrics

    def critic_regression(
        self,
        fast_critic_logits: torch.Tensor,
        slow_critic_logits: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regression loss for the critic: cross-entropy toward λ-returns + slow regularization.

        Args:
            fast_critic_logits: (..., bins)
            slow_critic_logits: (..., bins)
            lambda_returns: (...)

        Returns:
            (total_loss, return_loss, slow_loss) — all scalar tensors.
        """
        return_target = self.symlog_two_hot.encode(lambda_returns.detach())  # (..., bins)
        slow_probs = F.softmax(slow_critic_logits.detach(), dim=-1)          # (..., bins)

        # flatten all leading dims so cross_entropy sees (N, bins)
        fast_flat = fast_critic_logits.flatten(0, -2)
        target_flat = return_target.flatten(0, -2)
        slow_flat = slow_probs.flatten(0, -2)

        return_loss = F.cross_entropy(fast_flat, target_flat)
        slow_loss = F.cross_entropy(fast_flat, slow_flat)
        loss = return_loss + self.slow_regularization_weight * slow_loss

        return loss, return_loss, slow_loss

    def actor_loss(
        self,
        lambda_returns: torch.Tensor,
        values: torch.Tensor,
        dream_output: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_distribution = policy_distribution(dream_output["action_logits"])
        action_log_probs = action_distribution.log_prob(dream_output["actions"])
        entropy = action_distribution.entropy()

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
