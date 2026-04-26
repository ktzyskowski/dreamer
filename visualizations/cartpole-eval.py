#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "cartpole-eval.csv")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["step"], df["value"], linewidth=2)
ax.axhline(475, color="gray", linestyle="--", linewidth=1, label="475 target")
ax.set_title("CartPole-v1 Eval Performance (Dreamer)")
ax.set_xlabel("Gradient Step")
ax.set_ylabel("Mean Episode Return (100 episodes)")
ax.legend()
ax.annotate("Each point averaged over 100 episodes", xy=(1, 0), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9, color="gray",
            xytext=(0, -32), textcoords="offset points")

fig.tight_layout()
fig.savefig(HERE / "cartpole-eval.png", dpi=150)
print(f"Saved: {HERE / 'cartpole-eval.png'}")
