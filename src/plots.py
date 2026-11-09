"""Chart helpers for the per-park slides."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .park_effects import BB_TYPES

BB_LABELS = {"ground_ball": "Ground ball", "line_drive": "Line drive", "fly_ball": "Fly ball", "popup": "Popup"}
RED, BLUE = "#b03a2e", "#1f618d"


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)


def plot_park_bb(effects: pd.DataFrame, park: str, path=None, figsize=(8, 5), compact=False):
    """Bars of how a park treats pitchers on each batted-ball type, by batter side.

    compact=True makes a smaller version without a title, for placing on a slide.
    """
    e = effects[effects["park"] == park]
    if e.empty:
        return None
    types = [t for t in BB_TYPES if t in set(e["bb_type"])]
    x = np.arange(len(types))
    width = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    for i, (stand, color) in enumerate([("L", "#2e86c1"), ("R", "#e67e22")]):
        vals = [
            e.loc[(e["bb_type"] == t) & (e["stand"] == stand), "pitcher_edge"].mean() * 1000 for t in types
        ]
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=f"{stand}-handed batters")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BB_LABELS[t] for t in types], fontsize=9 if compact else 10)
    if compact:
        ax.set_ylabel("wOBA pts vs. league\n(+ helps pitcher)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=8)
    else:
        ax.set_ylabel("wOBA points on contact vs. league (positive = helps pitcher)")
        ax.set_title(f"{park}: effect on pitchers by batted-ball type")
        ax.legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_calibration(cal: pd.DataFrame, path=None):
    """Predicted vs. observed home-road gap by bin (backtest)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(cal["predicted_pts"].abs().max(), cal["observed_pts"].abs().max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="gray", linestyle="--", label="perfect prediction")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.plot(cal["predicted_pts"], cal["observed_pts"], "o-", color=BLUE, label="pitcher-seasons, grouped")
    ax.set_xlabel("Predicted home-minus-road gap (wOBA points on contact)")
    ax.set_ylabel("Observed home-minus-road gap (wOBA points on contact)")
    ax.set_title("Backtest: predicted vs. observed (positive = hitters did better at home)")
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
    colors = [RED if v < 0 else BLUE for v in shown["fit_pts"]]
    ax.barh(shown["archetype"], shown["fit_pts"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("How much better this park suits the pitch than it suits a typical pitch (wOBA points on contact)")
    ax.set_title(f"{park}: best and worst pitch archetypes, {'RHP' if hand == 'R' else 'LHP'}")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_park_archetypes_both(arch: pd.DataFrame, park: str, path=None, n: int = 3, figsize=(12.5, 2.7)):
    """Right-handed and left-handed pitchers side by side: the best and worst archetypes at a park."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    drew = False
    for ax, hand, label in zip(axes, ["R", "L"], ["Right-handed pitchers", "Left-handed pitchers"]):
        a = arch[(arch["park"] == park) & (arch["p_throws"] == hand)].sort_values("fit_pts")
        if a.empty:
            ax.axis("off")
            continue
        drew = True
        shown = pd.concat([a.head(n), a.tail(n)]).drop_duplicates("archetype")
        colors = [RED if v < 0 else BLUE for v in shown["fit_pts"]]
        ax.barh(shown["archetype"], shown["fit_pts"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("wOBA pts vs. a typical pitch (+ = better fit)", fontsize=8)
    if not drew:
        plt.close(fig)
        return None
    fig.tight_layout()
    _save(fig, path)
    return fig
