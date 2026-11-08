"""Chart helpers for the per-park slides."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .park_effects import BB_TYPES

BB_LABELS = {"ground_ball": "Ground ball", "line_drive": "Line drive", "fly_ball": "Fly ball", "popup": "Popup"}


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)


def plot_park_bb(effects: pd.DataFrame, park: str, path=None):
    """Bars of how a park treats pitchers on each batted-ball type, by batter side."""
    e = effects[effects["park"] == park]
    if e.empty:
        return None
    types = [t for t in BB_TYPES if t in set(e["bb_type"])]
    x = np.arange(len(types))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (stand, color) in enumerate([("L", "#2e86c1"), ("R", "#e67e22")]):
        vals = [
            e.loc[(e["bb_type"] == t) & (e["stand"] == stand), "pitcher_edge"].mean() * 1000 for t in types
        ]
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=f"{stand}-handed batters")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BB_LABELS[t] for t in types])
    ax.set_ylabel("wOBA points on contact vs. league (positive = helps pitcher)")
    ax.set_title(f"{park}: effect on pitchers by batted-ball type")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_park_archetypes(arch: pd.DataFrame, park: str, hand: str, path=None, n: int = 5):
    """Bars of the pitch archetypes that suit a park best and worst, for one pitcher hand."""
    a = arch[(arch["park"] == park) & (arch["p_throws"] == hand)]
    if a.empty:
        return None
    a = a.sort_values("fit_pts")
    shown = pd.concat([a.head(n), a.tail(n)]).drop_duplicates("archetype")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#b03a2e" if v < 0 else "#1f618d" for v in shown["fit_pts"]]
    ax.barh(shown["archetype"], shown["fit_pts"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("How much better this park suits the pitch than it suits a typical pitch (wOBA points on contact)")
    ax.set_title(f"{park}: best and worst pitch archetypes, {'RHP' if hand == 'R' else 'LHP'}")
    fig.tight_layout()
    _save(fig, path)
    return fig
