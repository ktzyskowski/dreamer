from modules.models.actor import Actor
from modules.models.critic import Critic
from modules.models.world_model import WorldModel
from modules.utils.buffer import ReplayBuffer


class Dreamer:
    def __init__(self, observation_space, action_space):
        self.world_model = WorldModel()
        self.actor = Actor()
        self.critic = Critic()

        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            recurrent_dim=128,
            capacity=10_000,
        )

    def train_world_model(self):
        pass

    def train_actor_critic(self):
        pass

    def save_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"

    def load_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"
