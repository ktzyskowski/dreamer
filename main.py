import argparse
import dataclasses
import logging
import sys

import torch

from src.config import Config, flatten, load_config
from src.data.buffer import ReplayBuffer
from src.env.factory import build_env
from src.rl.dreamer import Dreamer
from src.training.collector import Collector
from src.training.evaluator import Evaluator
from src.training.metrics import MetricsAggregator
from src.training.trainer import Trainer
from src.training.factory import LossFactory, ModelFactory
from src.util.torch_util import get_device


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-path", default="conf/config.yaml")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    config = load_config(Config, yaml_path=args.config_path)

    env = build_env(
        config.environment.type,
        name=config.environment.name,
        action_repeat=config.environment.action_repeat,
        extra_kwargs=config.environment.extra_kwargs,
    )
    eval_env = build_env(
        config.environment.type,
        name=config.environment.name,
        action_repeat=config.environment.action_repeat,
        extra_kwargs=config.environment.extra_kwargs,
    )
    metrics = MetricsAggregator(experiment_name=config.experiment_name)

    loss_factory = LossFactory(config)
    world_model_loss = loss_factory.new_world_model_loss()
    actor_critic_loss = loss_factory.new_actor_critic_loss()

    with metrics, env, eval_env:
        metrics.log_params(flatten(dataclasses.asdict(config)))

        observation_shape = env.observation_space.shape
        action_size = env.action_size

        replay_buffer = ReplayBuffer(
            observation_shape=observation_shape,
            action_size=action_size,
            capacity=config.replay_buffer.capacity,
            dtype=config.replay_buffer.dtype,
        )

        # ------------------------------------------------------------------- #

        if config.torch.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(config.torch.float32_matmul_precision)

        device = config.torch.device
        device = get_device(priority=device)
        # if device == "cuda" and not torch.cuda.is_available():
        #     logging.warning("cuda requested but not available, falling back to cpu")
        #     device = "cpu"

        model_factory = ModelFactory(config, observation_shape, action_size)
        encoder, decoder = model_factory.new_encoder_decoder()
        world_model, reward_predictor, continue_predictor = (
            model_factory.new_world_model()
        )
        actor, critic = model_factory.new_actor_critic()
        dreamer = Dreamer(
            encoder=encoder,
            decoder=decoder,
            world_model=world_model,
            actor=actor,
            critic=critic,
            reward_predictor=reward_predictor,
            continue_predictor=continue_predictor,
            dream_horizon=config.dreamer.dream_horizon,
        ).to(device)

        if device == "cuda":
            # compile the per-gradient-step hot paths, they contain tight
            # python loops over sequence_length and dream_horizon; compiling
            # collapses the per-step MLP + GRU kernels and cuts launch overhead.
            dreamer = torch.compile(dreamer)
            # dreamer.observe = torch.compile(dreamer.observe, dynamic=False)
            # dreamer.dream = torch.compile(dreamer.dream, dynamic=False)

        # ------------------------------------------------------------------- #

        collector = Collector(
            env=env,
            dreamer=dreamer,
            replay_buffer=replay_buffer,
            device=device,
        )

        evaluator = Evaluator(
            env=eval_env,
            dreamer=dreamer,
            device=device,
            n_episodes=config.training.n_eval_episodes,
        )

        trainer = Trainer(
            dreamer=dreamer,
            collector=collector,
            replay_buffer=replay_buffer,
            metrics=metrics,
            world_model_loss=world_model_loss,
            actor_critic_loss=actor_critic_loss,
            device=device,
            batch_size=config.training.batch_size,
            sequence_length=config.training.sequence_length,
            warmup_steps=config.training.warmup_steps,
            replay_ratio=config.training.replay_ratio,
            world_model_lr=config.world_model.learning_rate,
            actor_lr=config.actor.learning_rate,
            critic_lr=config.critic.learning_rate,
            grad_clip=config.training.grad_clip,
            checkpoint_dir=config.training.checkpoint_dir,
            save_every_n_gradient_steps=config.training.save_every_n_gradient_steps,
            evaluator=evaluator,
            eval_every_n_gradient_steps=config.training.eval_every_n_gradient_steps,
        )

        trainer.train(n_steps=config.training.n_steps)


if __name__ == "__main__":
    main()
