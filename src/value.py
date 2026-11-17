"""Rough value of a free agent: estimated wins above a replacement pitcher, and what each win would cost.

This is a back-of-envelope screen, not a WAR model. The assumptions are constants you can change:

  REPLACEMENT_EDGE  a replacement pitcher allows this much more wOBA than the league average
  WOBA_SCALE        wOBA points to runs per plate appearance (about 1.2)
  PA_BY_ROLE        plate appearances faced in a season (starter, reliever)
  RUNS_PER_WIN      runs that make one win

est_wins uses park-neutral talent (talent_woba), so it does not depend on where he signs.
cost_per_win_m is projected salary per estimated win ($M); lower means cheaper for what you get.
"""
import numpy as np
import pandas as pd

REPLACEMENT_EDGE = 0.015
WOBA_SCALE = 1.2
PA_BY_ROLE = {"S": 750, "R": 250}
RUNS_PER_WIN = 10.0


def add_value(summary: pd.DataFrame, salary: pd.DataFrame, league_woba: float) -> pd.DataFrame:
    """Add salary_m, est_wins, and cost_per_win_m to the free-agent summary (one row per name)."""
    out = summary.merge(salary[["name", "salary_m", "contract"]], on="name", how="left")
    repl = league_woba + REPLACEMENT_EDGE
    pa = out["role"].map(PA_BY_ROLE).fillna(PA_BY_ROLE["S"])
    out["est_wins"] = (repl - out["talent_woba"]) / WOBA_SCALE * pa / RUNS_PER_WIN
    out["cost_per_win_m"] = np.where((out["est_wins"] > 0.05) & out["salary_m"].notna(),
                                     out["salary_m"] / out["est_wins"], np.nan)
    return out


EDGES = [0.005, 0.010, 0.015, 0.020, 0.030]


def value_sensitivity(summary: pd.DataFrame, salary: pd.DataFrame, league_woba: float, edges=EDGES, top: int = 5) -> pd.DataFrame:
    """Cost-per-win rank of each pitcher under different replacement levels.

    Returns one row per pitcher with his rank (1 = cheapest per win) for each replacement edge, plus
    how many of the settings put him in the top `top`. Pitchers with no salary are left out.
    """
    global REPLACEMENT_EDGE
    keep = REPLACEMENT_EDGE
    out = None
    try:
        for e in edges:
            REPLACEMENT_EDGE = e
            v = add_value(summary, salary, league_woba)
            v = v[v["cost_per_win_m"].notna()].copy()
            v[f"rank_{int(round(e * 1000))}pts"] = v["cost_per_win_m"].rank(method="min").astype(int)
            v = v[["name", f"rank_{int(round(e * 1000))}pts"]]
            out = v if out is None else out.merge(v, on="name", how="outer")
    finally:
        REPLACEMENT_EDGE = keep
    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    out["times_in_top"] = (out[rank_cols] <= top).sum(axis=1)
    return out.sort_values(["times_in_top"] + rank_cols[:1], ascending=[False, True]).reset_index(drop=True)


def best_value_fits(fa_proj: pd.DataFrame, value: pd.DataFrame, min_wins: float = 1.0, n: int = 3) -> pd.DataFrame:
    """For each park, the starters most worth signing there: good enough, ranked by how well they fit the park.

    Eligible = starters (role S) with est_wins >= min_wins. fit_pts is the pitcher's edge at this park
    relative to a typical pitcher and to his own average (see src/fit.py), so Busch-type parks do not win
    everywhere. weak_fit marks parks where the top-n average fit is below 75% of the median across parks,
    meaning the fit is less distinct there than at most parks.
    """
    v = value[["name", "est_wins", "salary_m", "contract", "cost_per_win_m"]]
    p = fa_proj.drop(columns=[c for c in v.columns if c != "name" and c in fa_proj.columns]).merge(v, on="name", how="inner")
    p = p[(p["role"] == "S") & (p["est_wins"] >= min_wins)]
    top = (p.sort_values("fit_pts", ascending=False).groupby("park").head(n)
            .sort_values(["park", "fit_pts"], ascending=[True, False]))
    avg = top.groupby("park")["fit_pts"].mean()
    top["top_fit_avg"] = top["park"].map(avg)
    top["weak_fit"] = top["top_fit_avg"] < 0.75 * avg.median()
    return top[["park", "park_seasons", "name", "role", "fit_pts", "proj_woba", "talent_woba", "est_wins",
                "salary_m", "contract", "cost_per_win_m", "top_fit_avg", "weak_fit"]].reset_index(drop=True)
