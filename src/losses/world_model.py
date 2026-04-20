import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

from src.transforms.twohot import SymlogTwoHot, symlog
from src.util.probability import multi_categorical


class WorldModelLoss(nn.Module):
    """World model loss. Eq (2)/(3) in the DreamerV3 paper."""

    def __init__(
        self,
        n_categoricals: int,
        n_classes: int,
        two_hot_low: float,
        two_hot_high: float,
        two_hot_n_bins: int,
        beta_posterior: float = 0.1,
        beta_prior: float = 1.0,
        beta_prediction: float = 1.0,
        free_nats: float = 1.0,
    ):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.beta_posterior = beta_posterior
        self.beta_prior = beta_prior
        self.beta_prediction = beta_prediction
        self.free_nats = free_nats
        self.symlog_two_hot = SymlogTwoHot(
            low=two_hot_low, high=two_hot_high, n_bins=two_hot_n_bins
        )

    def prior_loss(self, observed_output: dict) -> torch.Tensor:
        """Compute dynamics loss from equation (3) in the DreamerV3 paper.

        Args:
            observed_output (dict): a dictionary with the outputs from the world model forward pass,
                used to calculate loss. Accesses `posterior_logits` and `prior_logits` keys.
        Returns:
            torch.Tensor: the dynamics loss (prior).
        """
        # Dynamics loss in Eq (3): KL(sg[posterior] || prior)
        posterior = multi_categorical(
            observed_output["posterior_logits"].detach(),
            self.n_categoricals,
            self.n_classes,
        )
        prior = multi_categorical(
            observed_output["prior_logits"], self.n_categoricals, self.n_classes
        )
        loss = kl_divergence(posterior, prior)
        return torch.clamp(loss, min=self.free_nats).mean()

    def posterior_loss(self, observed_output: dict) -> torch.Tensor:
        """Compute representation loss from equation (3) in the DreamerV3 paper.

        Args:
            observed_output (dict): a dictionary with the outputs from the world model forward pass,
                used to calculate loss. Accesses `posterior_logits` and `prior_logits` keys.
        Returns:
            torch.Tensor: the representation loss (posterior).
        """
        # Representation loss in Eq (3): KL(posterior || sg[prior])
        posterior = multi_categorical(
            observed_output["posterior_logits"], self.n_categoricals, self.n_classes
        )
        prior = multi_categorical(
            observed_output["prior_logits"].detach(),
            self.n_categoricals,
            self.n_classes,
        )
        loss = kl_divergence(posterior, prior)
        return torch.clamp(loss, min=self.free_nats).mean()

    def prediction_loss(
        self, batch: dict, observed_output: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prediction loss from DreamerV3 paper.

        Args:
            batch (dict): original sampled batch from replay buffer. Contains `observations`, `rewards`, `actions`, and `dones`.
            observed_output (dict): a dictionary with the outputs from the world model forward pass,
                used to calculate loss. Accesses `reconstructed_observations`, `predicted_continue_logits`, and
                `predicted_reward_logits` keys.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of (observation loss, continue_loss, reward_loss)
        """
        # Observation loss -------------------------------------------------- #

        observation_loss = F.mse_loss(
            symlog(batch["observations"]),
            observed_output["reconstructed_observations"],
        )

        # Continue loss ----------------------------------------------------- #

        predicted_continue_logits = observed_output[
            "predicted_continue_logits"
        ].squeeze(-1)
        continues = 1 - batch["dones"]
        continue_loss = F.binary_cross_entropy_with_logits(
            predicted_continue_logits, continues
        )

        # Reward loss ------------------------------------------------------- #

        reward_target = self.symlog_two_hot.encode(batch["rewards"])
        predicted_reward_logits = observed_output["predicted_reward_logits"]
        # cross_entropy expects class dim in position 1: (B, T, C) -> (B, C, T)
        reward_loss = F.cross_entropy(
            predicted_reward_logits.permute(0, 2, 1),
            reward_target.permute(0, 2, 1),
        )

        # ------------------------------------------------------------------- #

        return observation_loss, continue_loss, reward_loss

    def forward(
        self, batch: dict, observed_output: dict
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prior_loss = self.prior_loss(observed_output)
        posterior_loss = self.posterior_loss(observed_output)
        observation_loss, continue_loss, reward_loss = self.prediction_loss(
            batch, observed_output
        )

        prior_term = self.beta_prior * prior_loss
        posterior_term = self.beta_posterior * posterior_loss
        prediction_term = self.beta_prediction * (
            observation_loss + continue_loss + reward_loss
        )
        world_model_loss = prior_term + posterior_term + prediction_term

        metrics = {
            "world_model/prior_loss": prior_term,
            "world_model/posterior_loss": posterior_term,
            "world_model/prediction_loss": prediction_term,
            "world_model/observation_loss": observation_loss,
            "world_model/continue_loss": continue_loss,
            "world_model/reward_loss": reward_loss,
            "loss/world_model": world_model_loss,
        }
        return world_model_loss, metrics
