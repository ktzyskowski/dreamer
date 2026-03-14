import logging

import hydra
from omegaconf import DictConfig
import torch

from modules.models.world_model import WorldModel
from modules.nn.functions import count_parameters, get_device
from modules.utils.buffer import ReplayBuffer
from modules.utils.env import EnvironmentManager


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(config: DictConfig):
    # set up logging
    logging.basicConfig()

    # context manager automatically handles environment during training
    with EnvironmentManager(config) as env_manager:
        env = env_manager.env

        # log observation/action space information
        observation_shape = env.observation_space.shape
        action_size = env_manager.action_size
        logging.info("Observation space: %s", str(observation_shape))
        logging.info("Action size: %d", action_size)

        device = get_device()
        logging.info("Using device: %s", device)

        world_model = WorldModel(observation_shape=observation_shape, action_size=action_size, config=config)
        n_parameters = count_parameters(world_model)
        logging.info("World model # parameters: %d", n_parameters)

        # replay buffer will hold actual experience collected from environment
        replay_buffer = ReplayBuffer(
            observation_shape=observation_shape,
            action_shape=[action_size],
            # recurrent_dim=128,
            capacity=config.replay_buffer.capacity,
        )

        observation, _ = env.reset()

        action = torch.nn.functional.one_hot(torch.tensor(3), num_classes=action_size)

        logging.info("Example observation: %s", observation.shape)
        full_state, recurrent_state, posterior_log_probs, prior_log_probs = world_model.observed_step(
            observation, action
        )
        logging.info("Full state shape %s", full_state.shape)
        logging.info("Recurrent state shape %s", recurrent_state.shape)
        logging.info("Posterior shape %s", posterior_log_probs.shape)
        logging.info("Prior shape %s", prior_log_probs.shape)

        # for _ in range(100):
        #     # 1. collect experience
        #     for _ in range(config.replay_ratio):
        #         model_state = world_model.encode(observation, recurrent_state, no_grad=True)
        #         action = actor(model_state, no_grad=True)

        #         # take sampled action in environment, observe next observation and reward
        #         next_observation, reward, done = env.step(action)

        #         # add experience to replay buffer
        #         replay_buffer.add(observation, action, reward, done, recurrent_state)

        #         if done:
        #             observation = env.reset()
        #         else:
        #             observation = next_observation

        #     # 2. perform gradient update step
        #     batch = replay_buffer.sample(config.batch_size, config.sequence_length)


if __name__ == "__main__":
    main()
