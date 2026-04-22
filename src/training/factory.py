from src.config import Config, resolve_activation
from src.losses.actor_critic import ActorCriticLoss
from src.losses.world_model import WorldModelLoss
from src.nets.mlp import MultiLayerPerceptron
from src.rl.critic import DualCritic
from src.rl.world_model import WorldModel


class ModelFactory:
    def __init__(self, config: Config, observation_shape: tuple[int, ...], action_size: int):
        # calculate model shapes/sizes from configuration
        self.config = config
        self.latent_size = self.config.world_model.n_categoricals * self.config.world_model.n_classes
        self.recurrent_size = self.config.world_model.recurrent_net.recurrent_size
        self.full_state_size = self.recurrent_size + self.latent_size
        self.observation_shape = observation_shape
        self.action_size = action_size

    def new_encoder_decoder(self):
        encoder = MultiLayerPerceptron(
            input_dim=self.observation_shape[0],
            hidden_dims=self.config.world_model.encoder.mlp.hidden_dims,
            output_dim=self.latent_size,
            activation=resolve_activation(self.config.world_model.encoder.mlp.activation),
        )
        decoder = MultiLayerPerceptron(
            input_dim=self.full_state_size,
            hidden_dims=self.config.world_model.encoder.mlp.hidden_dims,
            output_dim=self.observation_shape[0],
            activation=resolve_activation(self.config.world_model.encoder.mlp.activation),
        )
        return encoder, decoder

    def new_world_model(self):
        world_model = WorldModel(
            input_size=self.latent_size,
            recurrent_size=self.recurrent_size,
            action_size=self.action_size,
            hidden_sizes=self.config.world_model.posterior_net.hidden_dims,
            n_categoricals=self.config.world_model.n_categoricals,
            n_classes=self.config.world_model.n_classes,
            activation=resolve_activation(self.config.world_model.posterior_net.activation),
            n_recurrent_blocks=self.config.world_model.recurrent_net.n_blocks,
        )
        reward_predictor = MultiLayerPerceptron(
            input_dim=self.full_state_size,
            hidden_dims=self.config.reward_predictor.hidden_dims,
            output_dim=self.config.two_hot.n_bins,
            activation=resolve_activation(self.config.reward_predictor.activation),
        )
        continue_predictor = MultiLayerPerceptron(
            input_dim=self.full_state_size,
            hidden_dims=self.config.continue_predictor.hidden_dims,
            output_dim=1,
            activation=resolve_activation(self.config.continue_predictor.activation),
        )
        return world_model, reward_predictor, continue_predictor

    def new_actor_critic(self):
        actor = MultiLayerPerceptron(
            input_dim=self.full_state_size,
            hidden_dims=self.config.actor.net.hidden_dims,
            output_dim=self.action_size,
            activation=resolve_activation(self.config.actor.net.activation),
            output_activation=None,
        )
        critic = DualCritic(
            input_dim=self.full_state_size,
            hidden_dims=self.config.critic.net.hidden_dims,
            output_dim=self.config.two_hot.n_bins,
            activation=resolve_activation(self.config.critic.net.activation),
            decay=self.config.critic.ema_decay,
        )
        return actor, critic


class LossFactory:
    def __init__(self, config: Config):
        self.config = config

    def new_world_model_loss(self):
        world_model_loss = WorldModelLoss(
            n_categoricals=self.config.world_model.n_categoricals,
            n_classes=self.config.world_model.n_classes,
            two_hot_low=self.config.two_hot.low,
            two_hot_high=self.config.two_hot.high,
            two_hot_n_bins=self.config.two_hot.n_bins,
            beta_posterior=self.config.world_model_loss.beta_posterior,
            beta_prior=self.config.world_model_loss.beta_prior,
            beta_prediction=self.config.world_model_loss.beta_prediction,
            free_nats=self.config.world_model_loss.free_nats,
        )
        return world_model_loss

    def new_actor_critic_loss(self):
        actor_critic_loss = ActorCriticLoss(
            two_hot_low=self.config.two_hot.low,
            two_hot_high=self.config.two_hot.high,
            two_hot_n_bins=self.config.two_hot.n_bins,
            discount=self.config.actor_critic_loss.discount,
            trace_decay=self.config.actor_critic_loss.trace_decay,
            entropy_coefficient=self.config.actor_critic_loss.entropy_coefficient,
            slow_regularization_weight=self.config.actor_critic_loss.slow_regularization_weight,
            advantage_norm_decay=self.config.actor_critic_loss.advantage_norm_decay,
        )
        return actor_critic_loss
