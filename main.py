import logging

import torch
import torch.nn as nn

from src.data.buffer import ReplayBuffer
from src.env.factory import build_env
from src.nets.activations import RMSNormSiLU
from src.nets.mlp import MultiLayerPerceptron
from src.rl.agent import Agent
from src.rl.dreamer import Dreamer
from src.rl.world_model import WorldModel
from src.training.collector import Collector
from src.training.metrics import MetricsAggregator
from src.training.trainer import Trainer


def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Env ---------------------------------------------------------------- #
    env = build_env("vector", name="CartPole-v1", action_repeat=1)
    observation_shape = (4,)
    action_size = 2

    # Hyperparameters ---------------------------------------------------- #
    recurrent_size = 128
    n_categoricals = 16
    n_classes = 16
    latent_size = n_categoricals * n_classes
    full_state_size = recurrent_size + latent_size
    hidden_dims = [128, 128]
    n_bins = 255
    activation = RMSNormSiLU

    dream_horizon = 15
    n_dreams = 64
    batch_size = 32
    sequence_length = 32
    warmup_steps = 1_024
    replay_ratio = 4

    # Models ------------------------------------------------------------- #
    # TODO: real MLP encoder/decoder for vector envs; CNN variants for pixels
    encoder: nn.Module = MultiLayerPerceptron(
        input_dim=observation_shape[0],
        hidden_dims=hidden_dims,
        output_dim=latent_size,
        activation=activation,
    )
    decoder: nn.Module = MultiLayerPerceptron(
        input_dim=full_state_size,
        hidden_dims=hidden_dims,
        output_dim=observation_shape[0],
        activation=activation,
    )

    world_model = WorldModel(
        input_size=latent_size,
        recurrent_size=recurrent_size,
        action_size=action_size,
        hidden_sizes=hidden_dims,
        n_categoricals=n_categoricals,
        n_classes=n_classes,
        activation=activation,
        n_recurrent_blocks=8,
    )

    agent = Agent(
        input_dim=full_state_size,
        hidden_dims=hidden_dims,
        action_size=action_size,
        n_bins=n_bins,
        activation=activation,
        critic_decay=0.98,
    )

    reward_predictor = MultiLayerPerceptron(
        input_dim=full_state_size,
        hidden_dims=hidden_dims,
        output_dim=n_bins,
        activation=activation,
    )
    continue_predictor = MultiLayerPerceptron(
        input_dim=full_state_size,
        hidden_dims=hidden_dims,
        output_dim=1,
        activation=activation,
    )

    dreamer = Dreamer(
        encoder=encoder,
        decoder=decoder,
        world_model=world_model,
        agent=agent,
        reward_predictor=reward_predictor,
        continue_predictor=continue_predictor,
        dream_horizon=dream_horizon,
        n_dreams=n_dreams,
    ).to(device)

    # Data & training infra --------------------------------------------- #
    replay_buffer = ReplayBuffer(
        observation_shape=observation_shape,
        action_size=action_size,
        capacity=1_000_000,
        dtype="float32",
    )

    collector = Collector(
        env=env,
        dreamer=dreamer,
        replay_buffer=replay_buffer,
        device=device,
    )

    trainer = Trainer(
        # dreamer=dreamer,
        # collector=collector,
        # replay_buffer=replay_buffer,
        # metrics=metrics,
        # batch_size=batch_size,
        # sequence_length=sequence_length,
        # warmup_steps=warmup_steps,
        # replay_ratio=replay_ratio,
        # device=device,
    )

    metrics = MetricsAggregator(experiment_name="dreamer")
    with metrics, env:
        # trainer.train(n_steps=10_000_000)
        pass


if __name__ == "__main__":
    main()
