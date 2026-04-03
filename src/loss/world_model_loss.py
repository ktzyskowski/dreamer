import torch
import torch.nn.functional as F

from src.data import ObservedOutput, WorldModelInput
from src.util.functions import symlog
from src.util.two_hot import TwoHot


class WorldModelLoss:
    # Eq (2) from paper

    def __init__(self, config):
        self.beta_posterior = config.world_model_loss.beta_posterior
        self.beta_prior = config.world_model_loss.beta_prior
        self.beta_prediction = config.world_model_loss.beta_prediction
        self.free_nats = config.world_model_loss.free_nats
        self.two_hot = TwoHot(low=config.two_hot.low, high=config.two_hot.high, n_bins=config.two_hot.n_bins)

    def calculate_kl_loss(self, input, target):
        loss = torch.nn.functional.kl_div(input, target=target, log_target=True, reduction="none")
        # sum KL over classes, per categorical
        loss = loss.sum(-1)
        # clip KL per categorical
        loss = torch.max(torch.full_like(loss, self.free_nats), loss)
        # sum over clipped categoricals, take mean per batch
        loss = loss.sum(-1).mean()
        loss = loss
        return loss

    def calculate_prior_loss(self, observed_output: ObservedOutput):
        posterior_log_probs = observed_output["posterior_log_probs"]
        prior_log_probs = observed_output["prior_log_probs"]

        # Dynamics loss in Eq (3)
        loss = self.calculate_kl_loss(
            prior_log_probs,
            target=posterior_log_probs.detach(),
        )
        loss = self.beta_prior * loss
        return loss

    def calculate_posterior_loss(self, observed_output: ObservedOutput):
        posterior_log_probs = observed_output["posterior_log_probs"]
        prior_log_probs = observed_output["prior_log_probs"]

        # Representation loss in Eq (3)
        loss = self.calculate_kl_loss(
            prior_log_probs.detach(),
            target=posterior_log_probs,
        )
        loss = self.beta_posterior * loss
        return loss

    def calculate_prediction_loss(self, batch: WorldModelInput, observed_output: ObservedOutput):
        # Prediction loss in Eq (3)

        observations = batch["observations"]
        reconstructed_observations = observed_output["reconstructed_observations"]
        observation_loss = F.mse_loss(observations, reconstructed_observations)

        dones = batch["dones"]
        predicted_continue_logits = observed_output["predicted_continue_logits"]
        continues = 1 - dones
        continue_loss = F.binary_cross_entropy_with_logits(predicted_continue_logits, continues)

        reward_logits = self.two_hot.encode(symlog(batch["rewards"]))
        predicted_reward_logits = observed_output["predicted_reward_logits"]
        # permute: (batch, sequence, classes) -> (batch, classes, sequence)
        # because cross_entropy expects class dimension in 2nd position
        reward_loss = F.cross_entropy(
            predicted_reward_logits.permute(0, 2, 1),
            reward_logits.permute(0, 2, 1),
        )

        loss = observation_loss + continue_loss + reward_loss
        loss = self.beta_prediction * loss
        return loss

    def __call__(self, batch: WorldModelInput, observed_output: ObservedOutput):
        prior_loss = self.calculate_prior_loss(observed_output)
        posterior_loss = self.calculate_posterior_loss(observed_output)
        prediction_loss = self.calculate_prediction_loss(batch, observed_output)
        loss = prior_loss + posterior_loss + prediction_loss
        return loss
