# Mastering Diverse Domains through World Models

[📄 Paper Source](https://arxiv.org/pdf/2301.04104)

This repository contains my implementation of the DreamerV3 paper, which was introduced by Hafner et al. in their paper _Mastering Diverse Domains through World Models_.

As always, I find that the best way to learn how something works is to take it apart and build it yourself!

## Setup

> This project utilizes the `uv` package manager, which you can find more information on [here](https://github.com/astral-sh/uv).

To install all the required dependencies, simply run the command below:

```zsh
uv sync
```

Currently, the code only works on my Windows PC. I suspect it has something to do with the underlying Atari emulators, since the rest of the code really only relies on PyTorch and NumPy, which work on most systems. YMMV.

Although the pipeline is non-existant at this point in the project, to run the main script you can run this command:

```zsh
uv run main.py --config-name=config
```

To get total LoC (out of curiosity) you can use this command:

```zsh
find . -name "*.py" -not -path "./.venv/*" | xargs wc -l | sort -rn
```

## Notes

### 1. Posterior/Prior KL Loss in Stochastic Start States

In environments with stochastic start states (such as in the Memory minigrid task in which the object cue is randomly selected), the prior has no way of correctly predicting the latent state given an initial zero recurrent tensor (opposed to the posterior which does have an observation). This is problematic because the prior has no clear learning signal, and the posterior is pulled towards the prior, so both are being negatively affected. The fix is simply to mask the KL loss when the recurrent state for the timestep is the initial zero tensor. This way, the noise is removed during loss calculation.
