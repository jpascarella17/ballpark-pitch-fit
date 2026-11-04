"""Chart helpers for the per-park slides."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_park_archetypes(effects: pd.DataFrame, park: str, hand: str, path=None, n: int = 5, min_bip: int = 100):
    """Bar chart of the best and worst pitch archetypes at one park for one handedness."""
    e = effects[(effects["park"] == park) & (effects["p_throws"] == hand) & (effects["bip"] >= min_bip)]
    if e.empty:
        return None
    e = e.sort_values("pitcher_edge")
    shown = pd.concat([e.head(n), e.tail(n)]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#b03a2e" if v < 0 else "#1f618d" for v in shown["pitcher_edge"]]
    ax.barh(shown["archetype"], shown["pitcher_edge"] * 1000, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("wOBA points on contact vs. league average for that archetype (positive = helps pitcher)")
    ax.set_title(f"{park}: best and worst pitch archetypes vs. {'RHP' if hand == 'R' else 'LHP'}")
    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    return fig
