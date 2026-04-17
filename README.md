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
