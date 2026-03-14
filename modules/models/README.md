# models/

This subpackage contains the main code for implementing DreamerV3. There are two main files, `actor.py` and `world_model.py`, which contain the `DiscreteActor` and `WorldModel` classes, respectively.

Most of the complexity is in `world_model.py`.

### actor.py

The `DiscreteActor` class simply wraps the `MultiLayerPerceptron` class found in `nn/mlp.py`, but adds extra functionality in the forward pass. The output layer is softmax-activated to produce action probabilities, and a uniform distribution is mixed in to prevent action probabilities of 0 (which can mess with KL loss calculations).

An action is also sampled from the probabilities and returned.

### world_model.py

This file is likely the largest in the entire codebase and arguably the most important since this is where the `WorldModel` class resides, which is the center of the DreamerV3 architecture. There are seven internal neural networks, which map to the paper's six as follows:

- **Encoder**: encodes observations for use within the world model. However, this component does not strictly correspond one-to-one with the Encoder from the paper. The paper's Encoder also includes the posterior net (included below) to output a distribution over stochastic latent states. Other implementations separate the Encoder into two components, and I thought that design was cleaner than how the paper lays things out, so I implement mine in this way as well*.

- **Decoder**: reconstructs observations from the full model state (h, z). This corresponds one-to-one with the Decoder in the paper.

- **Posterior net**: technically a component of the Encoder, as described in the paper. In this implementation, the posterior net models the distribution over stochastic latent states (z) WITH an observation.

- **Prior net**: models the distribution over stochastic latent states (z), WITHOUT an observation. This corresponds one-to-one with the Dynamics Predictor in the paper.

- **Reward predictor**: predicts rewards from the full model state. This corresponds one-to-one with the Reward Predictor in the paper.

- **Continue predictor**: predicts soft continuation labels from the full model state. This corresponds one-to-one with the Continue Predictor in the paper.

- **Recurrent model**: produces the next recurrent state, given the current reccurent state. This is called the Sequence Model in the paper.

> \* Another reason that I like this view of the internal world model components is that it explicitly differentiates between latent states that are generated _after_ a sample is observed (posterior) and latent states which are generated _before/without_ any samples (prior).
