import logging

import hydra
from omegaconf import DictConfig

from modules.dreamer import Dreamer
from modules.utils.buffer import ReplayBuffer
from modules.utils.env import EnvironmentManager


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    # set up logging
    logging.basicConfig()

    # context manager automatically handles environment during training
    with EnvironmentManager() as env:

        # log observation/action space information
        observation_space = env.observation_space
        action_space = env.action_space
        logging.info("Observation space: %s", str(observation_space))
        logging.info("Action space: %s", str(action_space))

        # dreamer class contains training code
        # dreamer = Dreamer(env)

        # replay buffer will hold actual experience collected from environment.
        replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            recurrent_dim=128,
            capacity=config.replay_buffer.capacity,
        )

        # observation, recurrent_state = env.reset(), None
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
