from torch import Tensor
from torch.distributions import OneHotCategoricalStraightThrough


class MultiCategorical:
    """Multi-categorical distribution."""

    def __init__(self, logits: Tensor, n_categoricals: int, n_classes: int):
        """Construct a new multi-categorical distribution.

        Args:
            logits (*, n_categoricals,  n_classes)
                 | (*, n_categoricals * n_classes): unnormalized probability logits.
            n_categoricals (int): number of categoricals.
            n_classes (int): number of classes per categorical.
        """
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes

        # reshape logits if flattened
        if logits.shape[-1] == n_categoricals * n_classes:
            logits = logits.reshape(*logits.shape[:-1], n_categoricals, n_classes)

        self.probs = logits.softmax(dim=-1)
        self.log_probs = self.probs.log()
        self.dist = OneHotCategoricalStraightThrough(probs=self.probs)

    def sample(self):
        """Sample a categorical state from the logits.

        The categorical and class dimension are flattened into a single feature dimension.

        Returns:
            state (*, n_categoricals * n_classes)
        """
        state = self.dist.sample()
        # flatten into single (n_categoricals * n_classes) dimension
        state = state.reshape(*state.shape[:-2], self.n_categoricals * self.n_classes)
        return state
