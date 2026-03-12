import torch

import torch.nn.functional as F


class WorldModelLoss:
    def __init__(
        self,
        beta_posterior=0.1,
        beta_prior=1.0,
        beta_prediction=1.0,
        free_nats=1.0,  # approx 1.44 bits
    ):
        self.beta_posterior = beta_posterior
        self.beta_prior = beta_prior
        self.beta_prediction = beta_prediction
        self.free_nats = free_nats

    def calculate_kl_loss(self, input, target):
        loss = torch.nn.functional.kl_div(
            input, target=target, log_target=True, reduction="none"
        )
        # sum KL over classes, per categorical
        loss = loss.sum(-1)
        # clip KL per categorical
        loss = torch.max(torch.full_like(loss, self.free_nats), loss)
        # sum over clipped categoricals, take mean per batch
        loss = loss.sum(-1).mean()
        loss = loss
        return loss

    def calculate_prior_loss(self, posterior_log_probs, prior_log_probs):
        loss = self.calculate_kl_loss(
            prior_log_probs,
            target=posterior_log_probs.detach(),
        )
        loss = self.beta_prior * loss
        return loss

    def calculate_posterior_loss(self, posterior_log_probs, prior_log_probs):
        loss = self.calculate_kl_loss(
            prior_log_probs.detach(),
            target=posterior_log_probs,
        )
        loss = self.beta_posterior * loss
        return loss

    def calculate_prediction_loss(
        self,
        observations,
        reconstructed_observations,
        rewards_twohot,
        predicted_reward_logits,
        dones,
        predicted_continue_logits,
    ):
        observation_loss = F.mse_loss(observations, reconstructed_observations)

        continues = 1 - dones
        continue_loss = F.binary_cross_entropy_with_logits(
            predicted_continue_logits, continues
        )

        # permute: (batch, sequence, classes) -> (batch, classes, sequence)
        # because cross_entropy expects class dimension in 2nd position
        reward_loss = F.cross_entropy(
            predicted_reward_logits.permute(0, 2, 1),
            rewards_twohot.permute(0, 2, 1),
        )

        loss = observation_loss + continue_loss + reward_loss
        return loss

    def __call__(self, batch):
        prior_loss = self.calculate_prior_loss(
            batch["posterior_log_probs"],
            batch["prior_log_probs"],
        )
        posterior_loss = self.calculate_posterior_loss(
            batch["posterior_log_probs"],
            batch["prior_log_probs"],
        )
        prediction_loss = self.calculate_prediction_loss(
            batch["observations"],
            batch["reconstructed_observations"],
            batch["rewards_twohot"],
            batch["predicted_reward_logits"],
            batch["dones"],
            batch["predicted_continue_logits"],
        )
        loss = prior_loss + posterior_loss + prediction_loss
        return loss
