"""Shared plotting helpers.

Kept deliberately small: consistent styling and figure saving, so that every
figure the README embeds looks like it came from the same project.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg

PALETTE = {"no_claim": "#4C72B0", "claim": "#C44E52", "baseline": "#937860"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "semibold"


def save_fig(fig, name: str) -> None:
    """Write a figure to reports/figures/<name>.png."""
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES / f"{name}.png", dpi=150)


def claim_rate_bar(df, col, ax, order=None, baseline=None):
    """Claim rate by category, with the base rate drawn as a reference line.

    The baseline line is the point of this chart: a bar is only interesting to
    the extent it sits away from the overall claim rate.
    """
    g = df.groupby(col, observed=True)[cfg.TARGET].agg(["size", "mean"])
    if order is not None:
        g = g.reindex(order)
    ax.bar([str(i) for i in g.index], g["mean"], color=PALETTE["claim"], alpha=0.85)
    if baseline is not None:
        ax.axhline(baseline, color=PALETTE["baseline"], ls="--", lw=1.4,
                   label=f"base rate {baseline:.3f}")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title(col, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    if max(len(str(i)) for i in g.index) > 8:
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([str(i) for i in g.index], rotation=20, ha="right")
    ax.set_ylabel("claim rate")
    ax.set_ylim(0, 1.05)
    for i, (n, m) in enumerate(zip(g["size"], g["mean"])):
        ax.text(i, m + 0.03, f"n={n}", ha="center", fontsize=7, color="#444")
    return ax
