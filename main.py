import logging
from dataclasses import asdict

import torch

from agent.actor import DiscreteActor
from env.factory import build_env
from rl.critic import DualCritic
from world_model import WorldModel
from src.training.trainer import Trainer
from src.training.metrics import MetricsAggregator
from data.buffer import ReplayBuffer
from src.util.env import EnvironmentManager

from src.config import Config, load_config
from src.util.config import count_parameters, flatten


def main(config: Config):
    logging.basicConfig()

    # faster performance on tensor cores if available
    if config.torch.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(config.torch.float32_matmul_precision)

    metrics = MetricsAggregator(experiment_name="dreamer")
    env = build_env("vector", name="CartPole-v1", action_repeat=1)
    with metrics, env:
        metrics.log_params(flatten(asdict(config)))
        # metrics.log_params({"world_model_parameters": count_parameters(...)})

    # metrics = MetricsAggregator(experiment_name="dreamer")
    # with metrics:
    #     metrics.log_params(flatten(asdict(config)))

    #     # context manager automatically handles environment during training
    #     with EnvironmentManager(config) as env:
    #         # extract observation/action sizes from environment
    #         action_size = env.action_size
    #         observation_shape = env.observation_space.shape
    #         assert observation_shape is not None

    #         replay_buffer = ReplayBuffer(observation_shape, action_size, config.replay_buffer.capacity, dtype=config.replay_buffer.dtype)
    #         world_model = WorldModel(observation_shape, action_size, config)
    #         actor = DiscreteActor(world_model.full_state_size, action_size, config)
    #         critic = DualCritic(world_model.full_state_size, config)
    #         logging.info("World model # parameters: %d", count_parameters(world_model))
    #         logging.info("Actor # parameters: %d", count_parameters(actor))
    #         logging.info("Critic # parameters: %d", count_parameters(critic))

    #         trainer = Trainer(env, world_model, actor, critic, replay_buffer, metrics, config)

    #         # load from checkpoint if specified
    #         start_step = 0
    #         start_gradient_step = 0
    #         if config.checkpoint_path is not None:
    #             checkpoint = trainer.checkpointer.load(config.checkpoint_path, trainer.device)
    #             start_step = checkpoint["step"]
    #             start_gradient_step = checkpoint["gradient_step"]
    #             logging.info("Resumed from checkpoint: %s (step=%d)", config.checkpoint_path, start_step)

    #         trainer.train(n_steps=10_000_000, start_step=start_step, start_gradient_step=start_gradient_step)


if __name__ == "__main__":
    main(load_config())
