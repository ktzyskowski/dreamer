from src.env.minigrid import MiniGridEnv
from src.env.vector import VectorEnv


def build_env(env_type: str, **kwargs):
    if env_type == "vector":
        return VectorEnv(**kwargs)
    elif env_type == "minigrid":
        return MiniGridEnv(**kwargs)
    elif env_type == "pixel":
        raise NotImplementedError
    raise ValueError("Invalid enviroment type.")
