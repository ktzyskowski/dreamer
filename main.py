import logging

import hydra
from omegaconf import DictConfig

from src.models.critic import Critic
from src.models.actor import DiscreteActor
from src.models.world_model import WorldModel

from src.util.functions import count_parameters
from src.util.buffer import ReplayBuffer
from src.util.env import EnvironmentManager
from src.trainer import Trainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    # set up logging
    logging.basicConfig()

    # context manager automatically handles environment during training
    with EnvironmentManager(config) as env:
        observation_shape = env.observation_space.shape
        action_size = env.action_size

        replay_buffer = ReplayBuffer(observation_shape, action_size, config)
        world_model = WorldModel(observation_shape, action_size, config)
        actor = DiscreteActor(320, action_size, config)
        critic = Critic(320, config)
        logging.info("World model # parameters: %d", count_parameters(world_model))
        logging.info("Actor # parameters: %d", count_parameters(actor))
        logging.info("Critic # parameters: %d", count_parameters(critic))

        trainer = Trainer(env, world_model, actor, critic, replay_buffer, config)
        # trainer.train(env, n_steps=256)


if __name__ == "__main__":
    main()
