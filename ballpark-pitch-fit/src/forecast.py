"""Calibrate the talent estimate on past seasons.

Question: when a pitcher has a bad (or good) season, how much of it carries over to the next one?
We build pitcher-season pairs (this season -> next season) and find the two numbers that predict
next season's park-neutral wOBA allowed best:

  luck_weight  share of a pitcher's contact luck to take out (0 = ignore it, 1 = trust expected wOBA fully)
  shrink_pa    plate appearances of league-average results to blend in (higher = pull harder toward average)

Each choice is judged on seasons it was not fitted on (leave one season-pair out), and compared with
simply using last season's raw or park-neutral wOBA allowed.
"""
import numpy as np
import pandas as pd

from .fit import pa_rows, summarize_pa

GAMMAS = [round(g, 2) for g in np.arange(0.0, 1.01, 0.1)]
SHRINKS = [50, 100, 200, 300, 500, 750, 1000, 1500, 2500]


def pitcher_seasons(df: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher-season: pa, raw_woba, neutral_woba, bip_rate, contact_luck_pts."""
    return summarize_pa(pa_rows(df, effects), ["pitcher", "game_year"]).reset_index()


def next_season_pairs(ps: pd.DataFrame, min_pa: int = 200) -> pd.DataFrame:
    """Pair each pitcher-season with the same pitcher's next season (columns ending _next)."""
    nxt = ps.copy()
    nxt["game_year"] = nxt["game_year"] - 1
    m = ps.merge(nxt, on=["pitcher", "game_year"], suffixes=("", "_next"))
    return m[(m["pa"] >= min_pa) & (m["pa_next"] >= min_pa)].reset_index(drop=True)


def _predict(m, gamma, k, league):
    signal = m["neutral_woba"] - gamma * m["contact_luck_pts"] / 1000
    return (signal * m["pa"] + league * k) / (m["pa"] + k)


def _sq_err(m, pred):
    """Squared error against next season's park-neutral wOBA, with the weight (next-season PA)."""
    return (m["neutral_woba_next"] - pred) ** 2, m["pa_next"]


def _wmse(m, pred):
    e, w = _sq_err(m, pred)
    return float(np.average(e, weights=w))


def _best_params(train):
    league = float(np.average(train["neutral_woba"], weights=train["pa"]))
    best = min(((g, k) for g in GAMMAS for k in SHRINKS),
               key=lambda gk: _wmse(train, _predict(train, gk[0], gk[1], league)))
    return best, league


def calibrate_talent(pairs: pd.DataFrame):
    """Fit luck_weight and shrink_pa and test them out of sample.

    Returns (summary, folds, params):
      summary  average prediction error (wOBA points, RMSE) for each way of predicting next season,
               pooled over the held-out season pairs
      folds    the held-out results and the parameters chosen for each fold
      params   luck_weight and shrink_pa fitted on all pairs (use these for the current projections)
    """
    if len(pairs) < 100:
        raise ValueError(f"Only {len(pairs)} pitcher-season pairs; lower min_pa in next_season_pairs.")
    models = ["Last season's raw wOBA allowed", "Last season's park-neutral wOBA allowed",
              "Park-neutral, shrunk toward average (500 PA)", "Calibrated (luck removed, shrunk)"]
    pooled = {m: [0.0, 0.0] for m in models}
    rows = []
    for held in sorted(pairs["game_year"].unique()):
        train, test = pairs[pairs["game_year"] != held], pairs[pairs["game_year"] == held]
        (g, k), league = _best_params(train)
        preds = {
            models[0]: test["raw_woba"],
            models[1]: test["neutral_woba"],
            models[2]: _predict(test, 0.0, 500, league),
            models[3]: _predict(test, g, k, league),
        }
        row = {"held_out_season": int(held), "pairs": len(test), "luck_weight": g, "shrink_pa": k}
        for name, pred in preds.items():
            e, w = _sq_err(test, pred)
            pooled[name][0] += float((e * w).sum())
            pooled[name][1] += float(w.sum())
            row[name] = float(np.sqrt(np.average(e, weights=w)) * 1000)
        rows.append(row)
    summary = pd.DataFrame({"model": models,
                            "rmse_pts": [float(np.sqrt(pooled[m][0] / pooled[m][1]) * 1000) for m in models]})
    (g, k), _ = _best_params(pairs)
    return summary, pd.DataFrame(rows), {"luck_weight": g, "shrink_pa": k}
