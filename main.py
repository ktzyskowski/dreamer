import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.models.actor import DiscreteActor
from src.models.critic import Critic
from src.models.world_model import WorldModel
from src.trainer import Trainer
from src.util.buffer import ReplayBuffer
from src.util.env import EnvironmentManager
from src.util.functions import count_parameters


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    # set up logging
    logging.basicConfig()
    wandb.init(
        project="dreamer",
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(_disable_stats=True),
        mode="offline",
    )

    # context manager automatically handles environment during training
    with EnvironmentManager(config) as env:
        # extract observation/action sizes from environment
        action_size = env.action_size
        observation_shape = env.observation_space.shape
        assert observation_shape is not None

        replay_buffer = ReplayBuffer(observation_shape, action_size, config)
        world_model = WorldModel(observation_shape, action_size, config)
        actor = DiscreteActor(world_model.full_state_size, action_size, config)
        critic = Critic(world_model.full_state_size, config)
        logging.info("World model # parameters: %d", count_parameters(world_model))
        logging.info("Actor # parameters: %d", count_parameters(actor))
        logging.info("Critic # parameters: %d", count_parameters(critic))

        trainer = Trainer(env, world_model, actor, critic, replay_buffer, config)
        trainer.train(env, n_steps=10_000)

    wandb.finish()


if __name__ == "__main__":
    main()
