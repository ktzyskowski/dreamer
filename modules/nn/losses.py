import torch


class WorldModelLoss:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, batch):
        # extract posterior/prior logits
        posterior_logits = batch["posterior_logits"]
        prior_logits = batch["prior_logits"]

        # dynamics: posterior is frozen, prior trained to match
        kl_dynamics = torch.nn.functional.kl_div(
            prior_logits, target=posterior_logits.detach()
        )
        # representation: prior is frozen, posterior trained to match
        kl_representation = torch.nn.functional.kl_div(
            prior_logits.detach(), target=posterior_logits
        )
