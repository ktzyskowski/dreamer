import logging

import torch
import torch.nn as nn

from src.config import Config, load_config, resolve_activation
from src.data.buffer import ReplayBuffer
from src.env.factory import build_env
from src.losses.actor_critic import ActorCriticLoss
from src.losses.world_model import WorldModelLoss
from src.nets.mlp import MultiLayerPerceptron
from src.rl.agent import Agent
from src.rl.dreamer import Dreamer
from src.rl.world_model import WorldModel
from src.training.collector import Collector
from src.training.evaluator import Evaluator
from src.training.metrics import MetricsAggregator
from src.training.trainer import Trainer


def main():
    logging.basicConfig(level=logging.INFO)

    cfg = load_config(Config)

    if cfg.torch.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)

    device = cfg.torch.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("cuda requested but not available, falling back to cpu")
        device = "cpu"

    # Env ---------------------------------------------------------------- #
    env = build_env(
        cfg.environment.type,
        name=cfg.environment.name,
        action_repeat=cfg.environment.action_repeat,
    )
    eval_env = build_env(
        cfg.environment.type,
        name=cfg.environment.name,
        action_repeat=cfg.environment.action_repeat,
    )
    observation_shape = (4,)
    action_size = 2

    # Derived sizes ------------------------------------------------------ #
    wm_cfg = cfg.world_model
    latent_size = wm_cfg.n_categoricals * wm_cfg.n_classes
    full_state_size = wm_cfg.recurrent_size + latent_size
    n_bins = cfg.two_hot.n_bins

    # Models ------------------------------------------------------------- #
    # TODO: CNN encoder/decoder variants for pixel envs
    encoder = MultiLayerPerceptron(
        input_dim=observation_shape[0],
        hidden_dims=wm_cfg.encoder.hidden_dims,
        output_dim=latent_size,
        activation=resolve_activation(wm_cfg.encoder.activation),
    )
    decoder = MultiLayerPerceptron(
        input_dim=full_state_size,
        hidden_dims=wm_cfg.decoder.hidden_dims,
        output_dim=observation_shape[0],
        activation=resolve_activation(wm_cfg.decoder.activation),
    )

    # posterior/prior share hidden_dims + activation with their configs;
    # WorldModel uses a single hidden_sizes/activation today, so pick from posterior_net
    world_model = WorldModel(
        input_size=latent_size,
        recurrent_size=wm_cfg.recurrent_size,
        action_size=action_size,
        hidden_sizes=wm_cfg.posterior_net.hidden_dims,
        n_categoricals=wm_cfg.n_categoricals,
        n_classes=wm_cfg.n_classes,
        activation=resolve_activation(wm_cfg.posterior_net.activation),
        n_recurrent_blocks=wm_cfg.recurrent_net.n_blocks,
    )

    agent = Agent(
        input_dim=full_state_size,
        hidden_dims=cfg.actor.net.hidden_dims,
        action_size=action_size,
        n_bins=n_bins,
        activation=resolve_activation(cfg.actor.net.activation),
        critic_decay=cfg.critic.ema_decay,
    )

    reward_predictor = MultiLayerPerceptron(
        input_dim=full_state_size,
        hidden_dims=wm_cfg.reward_predictor.hidden_dims,
        output_dim=n_bins,
        activation=resolve_activation(wm_cfg.reward_predictor.activation),
    )
    continue_predictor = MultiLayerPerceptron(
        input_dim=full_state_size,
        hidden_dims=wm_cfg.continue_predictor.hidden_dims,
        output_dim=1,
        activation=resolve_activation(wm_cfg.continue_predictor.activation),
    )

    dreamer = Dreamer(
        encoder=encoder,
        decoder=decoder,
        world_model=world_model,
        agent=agent,
        reward_predictor=reward_predictor,
        continue_predictor=continue_predictor,
        dream_horizon=cfg.dreamer.dream_horizon,
    ).to(device)

    if device == "cuda":
        # Compile the per-gradient-step hot paths. These contain tight
        # Python loops over sequence_length / dream_horizon; compiling
        # collapses the per-step MLP + GRU kernels and cuts launch overhead.
        dreamer.observe = torch.compile(dreamer.observe, dynamic=False)
        dreamer.dream = torch.compile(dreamer.dream, dynamic=False)

    # Data & training infra --------------------------------------------- #
    replay_buffer = ReplayBuffer(
        observation_shape=observation_shape,
        action_size=action_size,
        capacity=cfg.replay_buffer.capacity,
        dtype=cfg.replay_buffer.dtype,
    )

    wml_cfg = cfg.world_model_loss
    world_model_loss = WorldModelLoss(
        n_categoricals=wm_cfg.n_categoricals,
        n_classes=wm_cfg.n_classes,
        two_hot_low=cfg.two_hot.low,
        two_hot_high=cfg.two_hot.high,
        two_hot_n_bins=n_bins,
        beta_posterior=wml_cfg.beta_posterior,
        beta_prior=wml_cfg.beta_prior,
        beta_prediction=wml_cfg.beta_prediction,
        free_nats=wml_cfg.free_nats,
    )
    acl_cfg = cfg.actor_critic_loss
    actor_critic_loss = ActorCriticLoss(
        two_hot_low=cfg.two_hot.low,
        two_hot_high=cfg.two_hot.high,
        two_hot_n_bins=n_bins,
        discount=acl_cfg.discount,
        trace_decay=acl_cfg.trace_decay,
        entropy_coefficient=acl_cfg.entropy_coefficient,
        slow_regularization_weight=acl_cfg.slow_regularization_weight,
        advantage_norm_decay=acl_cfg.advantage_norm_decay,
    )

    metrics = MetricsAggregator(experiment_name="dreamer")

    tr_cfg = cfg.training
    with metrics, env, eval_env:
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
            n_episodes=tr_cfg.n_eval_episodes,
        )

        trainer = Trainer(
            dreamer=dreamer,
            collector=collector,
            replay_buffer=replay_buffer,
            metrics=metrics,
            world_model_loss=world_model_loss,
            actor_critic_loss=actor_critic_loss,
            device=device,
            batch_size=tr_cfg.batch_size,
            sequence_length=tr_cfg.sequence_length,
            warmup_steps=tr_cfg.warmup_steps,
            replay_ratio=tr_cfg.replay_ratio,
            action_repeat=cfg.environment.action_repeat,
            world_model_lr=wm_cfg.learning_rate,
            actor_lr=cfg.actor.learning_rate,
            critic_lr=cfg.critic.learning_rate,
            grad_clip=tr_cfg.grad_clip,
            checkpoint_dir=tr_cfg.checkpoint_dir,
            save_every_n_gradient_steps=tr_cfg.save_every_n_gradient_steps,
            evaluator=evaluator,
            eval_every_n_gradient_steps=tr_cfg.eval_every_n_gradient_steps,
        )

        trainer.train(n_steps=tr_cfg.n_steps)


if __name__ == "__main__":
    main()
