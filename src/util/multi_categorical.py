from torch import Tensor
from torch.distributions import OneHotCategoricalStraightThrough

from src.util.functions import mixin_uniform


class MultiCategorical:
    """Multi-Categorical distribution."""

    def __init__(
        self, logits: Tensor, n_categoricals: int, n_classes: int, unimix: float = 0.01
    ):
        """Construct a new multi-categorical distribution.

        Args:
            logits (*, n_categoricals,  n_classes)
                 | (*, n_categoricals * n_classes): unnormalized probability logits.
            n_categoricals (int): number of categoricals.
            n_classes (int): number of classes per categorical.
            unimix (float): percentage uniform distribution mixed in. 1 is fully uniform, 0 is no uniform mixin.
        """
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes

        # reshape logits if flattened
        if logits.shape[-1] == n_categoricals * n_classes:
            logits = logits.reshape(*logits.shape[:-1], n_categoricals, n_classes)

        self.probs = logits.softmax(dim=-1)
        self.probs = mixin_uniform(
            self.probs, split=unimix
        )  # encourages well-behaved KL loss
        self.log_probs = self.probs.log()
        self.dist = OneHotCategoricalStraightThrough(probs=self.probs)

    def sample(self):
        """Sample a categorical state from the logits.

        Returns:
            state (*, n_categoricals * n_classes)
        """
        state = self.dist.sample()
        # flatten into single (n_categoricals * n_classes) dimension
        state = state.reshape(*state.shape[:-2], self.n_categoricals * self.n_classes)
        return state
