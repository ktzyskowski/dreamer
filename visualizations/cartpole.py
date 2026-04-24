#!/usr/bin/env python3
"""Record a side-by-side comparison of a solved DreamerV3 CartPole agent vs. random."""

# currently broken, after I refactored main code

import sys
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import resolve_activation
from src.nets.mlp import MultiLayerPerceptron
from src.rl.agent import Agent
from src.rl.dreamer import Dreamer
from src.rl.world_model import WorldModel
from src.training.checkpoint import CheckpointManager

CHECKPOINT = "checkpoints/marked/checkpoint_007500.pt"
OUTPUT_VIDEO = "visualizations/cartpole_comparison.mp4"
DEVICE = "cpu"
FPS = 50

# Matches conf/config_cartpole.yaml exactly
RECURRENT_SIZE = 128
N_CATEGORICALS = 16
N_CLASSES = 16
HIDDEN_DIMS = [128, 128]
ACTIVATION = "rmsnorm_silu"
N_RECURRENT_BLOCKS = 1
N_BINS = 255
EMA_DECAY = 0.98
DREAM_HORIZON = 30


def build_dreamer(obs_size: int, action_size: int) -> Dreamer:
    latent_size = N_CATEGORICALS * N_CLASSES
    full_state_size = RECURRENT_SIZE + latent_size
    act = resolve_activation(ACTIVATION)

    return Dreamer(
        encoder=MultiLayerPerceptron(obs_size, HIDDEN_DIMS, latent_size, act),
        decoder=MultiLayerPerceptron(full_state_size, HIDDEN_DIMS, obs_size, act),
        world_model=WorldModel(
            input_size=latent_size,
            recurrent_size=RECURRENT_SIZE,
            action_size=action_size,
            hidden_sizes=HIDDEN_DIMS,
            n_categoricals=N_CATEGORICALS,
            n_classes=N_CLASSES,
            activation=act,
            n_recurrent_blocks=N_RECURRENT_BLOCKS,
        ),
        agent=Agent(full_state_size, HIDDEN_DIMS, action_size, N_BINS, act, EMA_DECAY),
        reward_predictor=MultiLayerPerceptron(
            full_state_size, HIDDEN_DIMS, N_BINS, act
        ),
        continue_predictor=MultiLayerPerceptron(full_state_size, HIDDEN_DIMS, 1, act),
        dream_horizon=DREAM_HORIZON,
    )


def run_dreamer_episode(dreamer: Dreamer) -> tuple[list[np.ndarray], float]:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    recurrent_state = torch.zeros(RECURRENT_SIZE, device=DEVICE)
    frames, total_reward = [], 0.0
    done = False

    while not done:
        frames.append(env.render())
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        action, recurrent_state = dreamer.act(obs_t, recurrent_state, greedy=True)
        obs, reward, terminated, truncated, _ = env.step(action.argmax().item())
        done = terminated or truncated
        total_reward += reward

    frames.append(env.render())
    env.close()
    return frames, total_reward


def run_random_episode(action_size: int) -> tuple[list[np.ndarray], float]:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    frames, total_reward = [], 0.0
    done = False

    while not done:
        frames.append(env.render())
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        total_reward += reward

    frames.append(env.render())
    env.close()
    return frames, total_reward


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    bar = Image.new("RGB", (frame.shape[1], 32), color=(30, 30, 30))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 7), text, fill=(220, 220, 220), font=font)
    return np.concatenate([frame, np.array(bar)], axis=0)


def pad_to(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    while len(frames) < length:
        frames.append(frames[-1])
    return frames


def save_video(
    dreamer_frames: list[np.ndarray],
    random_frames: list[np.ndarray],
    dreamer_reward: float,
    random_reward: float,
    path: str,
) -> None:
    n = max(len(dreamer_frames), len(random_frames))
    dreamer_frames = pad_to(dreamer_frames, n)
    random_frames = pad_to(random_frames, n)

    with imageio.get_writer(path, fps=FPS, macro_block_size=None) as writer:
        for df, rf in zip(dreamer_frames, random_frames):
            left = add_label(df, f"DreamerV3  score: {int(dreamer_reward)}")
            right = add_label(rf, f"Random  score: {int(random_reward)}")
            writer.append_data(np.concatenate([left, right], axis=1))


def main() -> None:
    probe = gym.make("CartPole-v1")
    obs_size = probe.observation_space.shape[0]
    action_size = probe.action_space.n
    probe.close()

    dreamer = build_dreamer(obs_size, action_size).to(DEVICE)
    dreamer.eval()

    checkpointer = CheckpointManager(
        directory=str(Path(CHECKPOINT).parent),
        modules={"dreamer": dreamer},
    )
    meta = checkpointer.load(CHECKPOINT, device=DEVICE)
    print(
        f"Loaded checkpoint — env step {meta['step']}, gradient step {meta['gradient_step']}"
    )

    print("Running DreamerV3 episode...")
    dreamer_frames, dreamer_reward = run_dreamer_episode(dreamer)
    print(f"  Score: {dreamer_reward}")

    print("Running random episode...")
    random_frames, random_reward = run_random_episode(action_size)
    print(f"  Score: {random_reward}")

    out = Path(OUTPUT_VIDEO)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_video(dreamer_frames, random_frames, dreamer_reward, random_reward, str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
