class Dreamer:
    def __init__(self, world_model, actor, critic, replay_buffer, env):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.env = env

    def collect_samples(self):
        pass

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
