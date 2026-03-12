import torch


class WorldModelLoss:
    def __init__(self, beta_posterior=1.0, beta_prior=1.0, free_bits=1.0):
        self.beta_posterior = beta_posterior
        self.beta_prior = beta_prior
        self.free_nats = 1.0

    def __call__(self, batch):
        # extract posterior/prior logits
        posterior_log_probs = batch["posterior_log_probs"]
        prior_log_probs = batch["prior_log_probs"]

        # prior loss
        kl_prior = torch.nn.functional.kl_div(
            prior_log_probs,
            target=posterior_log_probs.detach(),
            log_target=True,
        )
        kl_prior = torch.max(
            torch.full_like(kl_prior, self.free_nats),
            kl_prior,
        )
        kl_prior = self.beta_prior * kl_prior

        # posterior loss
        kl_posterior = torch.nn.functional.kl_div(
            prior_log_probs.detach(),
            target=posterior_log_probs,
            log_target=True,
        )
        kl_posterior = torch.max(
            torch.full_like(kl_posterior, self.free_nats),
            kl_posterior,
        )
        kl_posterior = self.beta_posterior * kl_posterior
