import logging

import hydra
from omegaconf import DictConfig
import torch

from src.models.actor import DiscreteActor
from src.models.world_model import WorldModel
from src.nn.functions import count_parameters, get_device
from src.buffer import ReplayBuffer
from src.env import EnvironmentManager


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(config: DictConfig):
    # set up logging
    logging.basicConfig()

    # context manager automatically handles environment during training
    with EnvironmentManager(config) as env:

        # log observation/action space information
        observation_shape = env.observation_space.shape
        action_size = env.action_size
        logging.info("Observation space: %s", str(observation_shape))
        logging.info("Action size: %d", action_size)

        device = get_device()
        logging.info("Using device: %s", device)

        # world model
        world_model = WorldModel(
            observation_shape=observation_shape,
            action_size=action_size,
            config=config,
        ).to(device)
        logging.info("World model # parameters: %d", count_parameters(world_model))

        # actor
        actor = DiscreteActor(
            input_dim=world_model.full_state_size,
            hidden_dims=[128, 128],
            action_dim=action_size,
        ).to(device)
        logging.info("Actor # parameters: %d", count_parameters(actor))

        # replay buffer will hold actual experience collected from environment
        replay_buffer = ReplayBuffer(
            observation_shape=observation_shape,
            action_shape=[action_size],
            capacity=config.replay_buffer.capacity,
        )

        # observation = env.reset()
        # observation, reward, done = env.step(torch.tensor(2))

        # action = torch.nn.functional.one_hot(torch.tensor(3), num_classes=action_size)

        # logging.info("Example observation: %s", observation.shape)
        # full_state, recurrent_state, posterior_log_probs, prior_log_probs = world_model.observed_step(
        #     observation, action
        # )
        # logging.info("Full state shape %s", full_state.shape)
        # logging.info("Recurrent state shape %s", recurrent_state.shape)
        # logging.info("Posterior shape %s", posterior_log_probs.shape)
        # logging.info("Prior shape %s", prior_log_probs.shape)

        collect_experience(
            actor,
            world_model,
            env,
            replay_buffer,
            n_steps=256,
        )

        # 1. collect experience into replay buffer

        # 2. sample experience from replay buffer

        # 3. train world model

        # 4. train actor/critic


@torch.no_grad()
def collect_experience(actor, world_model, env, replay_buffer, n_steps: int = 256):
    assert actor.device == world_model.device, "Actor and world model are on different devices."
    device = actor.device

    recurrent_state = torch.zeros(world_model.recurrent_size, device=device)
    observation = env.reset()

    for _ in range(n_steps):
        full_state = world_model.get_full_state(observation.to(device), recurrent_state)
        action, _ = actor(full_state)
        action_idx = action.argmax(dim=-1).cpu().item()

        next_observation, reward, done = env.step(action_idx)
        replay_buffer.add(observation, action, reward, done)

        if done:
            recurrent_state = torch.zeros(world_model.recurrent_size, device=device)
            observation, _ = env.reset()
        else:
            recurrent_state = world_model.get_next_recurrent_state(full_state, action, recurrent_state)
            observation = next_observation


if __name__ == "__main__":
    main()
