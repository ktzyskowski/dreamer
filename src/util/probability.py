import torch
import torch.nn.functional as F
from torch.distributions import (
    Distribution,
    Independent,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
)


def multi_categorical(
    logits: torch.Tensor, n_categoricals: int, n_classes: int
) -> Distribution:
    """Build a multi-categorical distribution from (possibly flat) logits.

    The returned distribution has event_shape (n_categoricals, n_classes).

    Args:
        logits (*, n_categoricals, n_classes) | (*, n_categoricals * n_classes): unnormalized probability logits.
        n_categoricals (int): number of categoricals.
        n_classes (int): number of classes per categorical.
    """
    if logits.shape[-1] == n_categoricals * n_classes:
        logits = logits.unflatten(-1, (n_categoricals, n_classes))
    return Independent(OneHotCategoricalStraightThrough(logits=logits), 1)


def policy_distribution(
    logits: torch.Tensor, uniform_mix: float = 0.01
) -> Distribution:
    """Build a one-hot categorical policy distribution from action logits.

    A small uniform mixture is blended into the probabilities to keep every action's
    log-prob finite, which prevents gradient collapse when an action probability would
    otherwise round to zero.

    Args:
        logits (*, n_actions): unnormalized action logits from the actor.
        uniform_mix (float): fraction of mass assigned to the uniform.
    """
    probs = mixin_uniform(F.softmax(logits, dim=-1), split=uniform_mix, dim=-1)
    return OneHotCategorical(probs=probs)


def mixin_uniform(probs: torch.Tensor, split=0.01, dim=-1) -> torch.Tensor:
    """Mixes a uniform distribution with the given probabilities.

    For a probability distribution with N outcomes, the mixed distribution will have probabilities:

    p_i = (1 - split)*p_i + split/N

    Args:
        probs (*): the tensor of probabilities.
        split (float): percentage assigned to uniform distribution.
        dim (int): tensor probability dimension.
    """
    # create uniform distribution over specified dimension
    uniform = torch.ones_like(probs) / probs.shape[dim]
    # mix uniform with given distribution
    mixed = (1 - split) * probs + split * uniform
    return mixed
