import hydra
import gymnasium as gym
from omegaconf import DictConfig
from modules.utils.buffer import ReplayBuffer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    env = gym.make(config.environment.name)

    # instantiate replay buffer for experience
    replay_buffer = ReplayBuffer(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=config.replay_buffer.capacity,
    )

    # release resources
    env.close()


if __name__ == "__main__":
    main()
