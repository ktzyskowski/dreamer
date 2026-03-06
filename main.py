import hydra
from omegaconf import DictConfig, OmegaConf

from modules.utils.buffer import ReplayBuffer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    replay_buffer = ReplayBuffer(
        observation_shape=(3, 64, 64),
        action_shape=(3,),
        capacity=10_000,
    )

    print(replay_buffer.capacity)


if __name__ == "__main__":
    main()
