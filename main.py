import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from modules.utils.buffer import ReplayBuffer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    replay_buffer = ReplayBuffer(**cfg.replay_buffer)

    for _ in range(500):
        replay_buffer.add(
            observation=np.random.rand(*cfg.replay_buffer.observation_shape).astype(np.float32),
            action=np.random.rand(*cfg.replay_buffer.action_shape).astype(np.float32),
            reward=np.random.rand(),
            next_observation=np.random.rand(*cfg.replay_buffer.observation_shape).astype(np.float32),
            done=np.random.choice([0.0, 1.0]),
        )

    print(replay_buffer.capacity)

    batch = replay_buffer.sample(batch_size=32, sequence_length=8)
    print(batch["observations"].shape)
    print(batch["actions"].shape)
    print(batch["rewards"].shape)
    print(batch["dones"].shape)


if __name__ == "__main__":
    main()
