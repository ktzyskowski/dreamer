from modules.models.actor import Actor
from modules.models.critic import Critic
from modules.models.world_model import WorldModel


class Dreamer:
    def __init__(self):
        self.world_model = WorldModel()
        self.actor = Actor()
        self.critic = Critic()

    def train(self):
        pass

    def save_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"

    def load_checkpoint(self, path):
        if not path.endswith(".pth"):
            path += ".pth"
