import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import torch

from src.models.actor import DiscreteActor
from src.models.critic import DualCritic
from src.models.world_model import WorldModel
from src.trainer import Trainer
from src.util.buffer import ReplayBuffer
from src.util.env import EnvironmentManager
from src.util.functions import count_parameters, flatten
from src.util.checkpoint import load_checkpoint

import os


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    # create checkpoint folder if not exists
    os.makedirs("checkpoints", exist_ok=True)

    # faster performance on tensor cores if available
    if "float32_matmul_precision" in config.torch:
        torch.set_float32_matmul_precision(config.torch.float32_matmul_precision)

    # set up logging
    logging.basicConfig()
    mlflow.set_experiment("dreamer")
    mlflow.start_run(log_system_metrics=True)
    mlflow.log_params(flatten(OmegaConf.to_container(config, resolve=True)))

    # context manager automatically handles environment during training
    with EnvironmentManager(config) as env:
        # extract observation/action sizes from environment
        action_size = env.action_size
        observation_shape = env.observation_space.shape
        assert observation_shape is not None

        replay_buffer = ReplayBuffer(observation_shape, action_size, config.replay_buffer.capacity, dtype=config.replay_buffer.dtype)
        world_model = WorldModel(observation_shape, action_size, config)
        actor = DiscreteActor(world_model.full_state_size, action_size, config)
        critic = DualCritic(world_model.full_state_size, config)
        logging.info("World model # parameters: %d", count_parameters(world_model))
        logging.info("Actor # parameters: %d", count_parameters(actor))
        logging.info("Critic # parameters: %d", count_parameters(critic))

        trainer = Trainer(env, world_model, actor, critic, replay_buffer, config)

        # load from checkpoint if specified
        start_step = 0
        start_gradient_step = 0
        if "checkpoint_path" in config and config.checkpoint_path is not None:
            checkpoint = load_checkpoint(
                config.checkpoint_path,
                world_model,
                actor,
                critic,
                trainer.world_model_optimizer,
                trainer.actor_optimizer,
                trainer.critic_optimizer,
                trainer.device,
            )
            start_step = checkpoint["step"]
            start_gradient_step = checkpoint["gradient_step"]
            logging.info("Resumed from checkpoint: %s (step=%d)", config.checkpoint_path, start_step)

        trainer.train(env, n_steps=10_000_000, start_step=start_step, start_gradient_step=start_gradient_step)

    mlflow.end_run()


if __name__ == "__main__":
    main()
