import logging

import hydra
from omegaconf import DictConfig
import torch

from src.models.actor import DiscreteActor
from src.models.world_model import WorldModel
from src.nn.functions import count_parameters, get_device
from src.buffer import ReplayBuffer
from src.env import EnvironmentManager
from src.trainer import Trainer


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(config: DictConfig):
    # set up logging
    logging.basicConfig()

    # context manager automatically handles environment during training
    with EnvironmentManager(config) as env:
        observation_shape = env.observation_space.shape
        action_size = env.action_size

        replay_buffer = ReplayBuffer(
            observation_shape=observation_shape,
            action_shape=(action_size,),
            capacity=config.replay_buffer.capacity,
        )

        world_model = WorldModel(
            observation_shape=observation_shape,
            action_size=action_size,
            config=config,
        )
        logging.info("World model # parameters: %d", count_parameters(world_model))
        actor = DiscreteActor(
            input_dim=world_model.full_state_size,
            hidden_dims=[128, 128],
            action_dim=action_size,
        )
        logging.info("Actor # parameters: %d", count_parameters(actor))

        trainer = Trainer(world_model, actor, replay_buffer, config)

        # 1. collect experience into replay buffer
        trainer.collect_experience(env)

        # 2. sample experience from replay buffer

        # 3. train world model

        # 4. train actor/critic


if __name__ == "__main__":
    main()
