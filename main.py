import hydra

from omegaconf import DictConfig
from modules.env import EnvironmentContext
from modules.utils.buffer import ReplayBuffer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):

    with EnvironmentContext() as env:
        observation_space = env.observation_space
        action_space = env.action_space
        print(observation_space)
        print(action_space)

        # create replay buffer
        replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            capacity=config.replay_buffer.capacity,
        )

        observation, _ = env.reset()
        print(observation.shape)


if __name__ == "__main__":
    main()
