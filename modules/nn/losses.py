from torch import nn

from modules.nn.functions import deconstruct_batch


class WorldModelLoss(nn.Module):
    # Eq (2)
    def __init__(
        self,
        world_model,
        prediction_beta=1.0,
        dynamics_beta=1.0,
        representation_beta=0.1,
    ):
        super().__init__()
        self.prediction_loss = WorldModelPredictionLoss(world_model)
        self.prediction_beta = prediction_beta
        self.dynamics_loss = WorldModelDynamicsLoss(world_model)
        self.dynamics_beta = dynamics_beta
        self.representation_loss = WorldModelRepresentationLoss(world_model)
        self.representation_beta = representation_beta

    def forward(self, batch):
        batch_prediction_loss = self.prediction_loss(batch)
        batch_dynamics_loss = self.dynamics_loss(batch)
        batch_representation_loss = self.representation_loss(batch)
        return (
            self.prediction_beta * batch_prediction_loss
            + self.dynamics_beta * batch_dynamics_loss
            + self.representation_beta * batch_representation_loss
        )


class WorldModelPredictionLoss(nn.Module):
    # Eq (3)
    def __init__(self, world_model):
        super().__init__()
        self.world_model = world_model

    def forward(self, batch):
        pass


class WorldModelDynamicsLoss(nn.Module):
    # Eq (3)
    def __init__(self, world_model):
        super().__init__()
        self.world_model = world_model

    def forward(self, batch):
        pass


class WorldModelRepresentationLoss(nn.Module):
    # Eq (3)
    def __init__(self, world_model):
        super().__init__()
        self.world_model = world_model

    def forward(self, batch):
        pass
