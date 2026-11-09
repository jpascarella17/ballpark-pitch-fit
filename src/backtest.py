"""Backtest: do park effects predict how pitchers did at home versus on the road?

For every pitcher-season:
  observed gap   mean (actual - expected) wOBA on contact at his home park minus on the road
  predicted gap  the same difference, using park effects estimated ONLY from other seasons
                 (leave-one-season-out), so the model never sees the season it is predicting
  baseline       the same again, but with one park-wide number per park instead of effects by
                 batter side and batted-ball type

A positive gap means hitters did better at the pitcher's home park than on the road. If the
batted-ball model beats the park-wide baseline, the pitcher-specific part adds real information.
"""
import numpy as np
import pandas as pd

from .park_effects import park_bb_effects


def add_home_flag(contact: pd.DataFrame, side: pd.DataFrame) -> pd.DataFrame:
    """Add is_home to each ball in play.

    `side` needs game_pk, pitcher, and inning_topbot (from the raw Statcast files). The home team
    pitches in the top half of the inning, so a pitcher is at home when inning_topbot == "Top".
    """
    s = side.drop_duplicates(["game_pk", "pitcher"]).copy()
    s["is_home"] = s["inning_topbot"].eq("Top")
    return contact.merge(s[["game_pk", "pitcher", "is_home"]], on=["game_pk", "pitcher"], how="inner")


def attach_loso_effects(contact: pd.DataFrame, shrink_k: int = 200) -> pd.DataFrame:
    """Predicted park effect for every ball in play, from the other seasons only.

    eff_full  effect for (park, batter side, batted-ball type)
    eff_park  one park-wide effect (average gap at the park minus the league average)
    Balls in play at parks with no data in the other seasons are dropped.
    """
    parts = []
    for season in sorted(contact["game_year"].unique()):
        train = contact[contact["game_year"] != season]
        test = contact[contact["game_year"] == season].copy()
        eff = park_bb_effects(train, shrink_k)[["park", "stand", "bb_type", "effect"]]
        eff = eff.rename(columns={"effect": "eff_full"})
        test = test.merge(eff, on=["park", "stand", "bb_type"], how="left")
        park_only = train.groupby("park")["woba_gap"].mean() - train["woba_gap"].mean()
        test["eff_park"] = test["park"].map(park_only)
        parts.append(test)
    out = pd.concat(parts, ignore_index=True)
    n0 = len(out)
    out = out.dropna(subset=["eff_full", "eff_park"])
    print(f"{len(out):,} balls in play kept ({n0 - len(out):,} dropped: park has no data in the other seasons)")
    return out


def home_road_table(test: pd.DataFrame, min_bip: int = 100) -> pd.DataFrame:
    """One row per pitcher-season with observed and predicted home-minus-road gaps (in wOBA)."""
    g = (test.groupby(["pitcher", "game_year", "is_home"])
             .agg(bip=("woba_gap", "size"), gap=("woba_gap", "mean"),
                  full=("eff_full", "mean"), park=("eff_park", "mean"))
             .unstack("is_home"))
    g.columns = [f"{a}_{'home' if b else 'road'}" for a, b in g.columns]
    g = g.dropna()
    g = g[(g["bip_home"] >= min_bip) & (g["bip_road"] >= min_bip)].reset_index()
    g["obs"] = g["gap_home"] - g["gap_road"]
    g["pred_full"] = g["full_home"] - g["full_road"]
    g["pred_park"] = g["park_home"] - g["park_road"]
    g["extra"] = g["pred_full"] - g["pred_park"]            # the pitcher-specific part
    g["weight"] = 1 / (1 / g["bip_home"] + 1 / g["bip_road"])  # more balls in play = more reliable
    return g


def _wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def _wls_r2(y, X, w):
    """Weighted least squares of y on X (with intercept): returns weighted R^2 and coefficients."""
    X1 = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X1 * sw[:, None], y * sw, rcond=None)
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - X1 @ beta) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return 1 - ss_res / ss_tot, beta


def evaluate(table: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Compare the batted-ball model with the park-factor baseline on the home-road gap.

    corr_park / corr_full   weighted correlation of the prediction with the observed gap
    r2_park                 weighted R^2 of the park-factor baseline
    r2_both                 R^2 after adding the pitcher-specific part (full model minus baseline)
    r2_gain                 r2_both - r2_park, with a 90% bootstrap interval (r2_gain_lo, r2_gain_hi)
    slope_full              slope of observed on predicted (1.0 means the effect size is right)
    extra_weight            weight on the pitcher-specific part (0 = nothing, 1 = exactly right), 90% interval
    extra_sd_pts            how much the pitcher-specific part varies between pitchers (wOBA points); next to
                            obs_sd_pts (variation in the observed gap) it shows how little a test could detect
    """
    if len(table) < 30:
        raise ValueError(f"Only {len(table)} pitcher-seasons in the table; lower min_bip in home_road_table "
                         "or check that the home/road flag was attached.")
    y = table["obs"].to_numpy(float)
    w = table["weight"].to_numpy(float)
    park = table["pred_park"].to_numpy(float)
    full = table["pred_full"].to_numpy(float)
    extra = table["extra"].to_numpy(float)

    r2_park, _ = _wls_r2(y, park[:, None], w)
    r2_both, b_both = _wls_r2(y, np.column_stack([park, extra]), w)
    _, b_full = _wls_r2(y, full[:, None], w)

    rng = np.random.default_rng(seed)
    gains, corr_gains, weights = [], [], []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, _ = _wls_r2(y[i], park[i, None], w[i])
        b, beta = _wls_r2(y[i], np.column_stack([park[i], extra[i]]), w[i])
        gains.append(b - a)
        weights.append(beta[2])
        corr_gains.append(_wcorr(full[i], y[i], w[i]) - _wcorr(park[i], y[i], w[i]))

    return {
        "n": n,
        "corr_park": _wcorr(park, y, w),
        "corr_full": _wcorr(full, y, w),
        "corr_gain_lo": float(np.percentile(corr_gains, 5)),
        "corr_gain_hi": float(np.percentile(corr_gains, 95)),
        "r2_park": r2_park,
        "r2_both": r2_both,
        "r2_gain": r2_both - r2_park,
        "r2_gain_lo": float(np.percentile(gains, 5)),
        "r2_gain_hi": float(np.percentile(gains, 95)),
        "slope_full": float(b_full[1]),
        # weight on the pitcher-specific part: 0 = it does nothing, 1 = the model's size is exactly right
        "extra_weight": float(b_both[2]),
        "extra_weight_lo": float(np.percentile(weights, 5)),
        "extra_weight_hi": float(np.percentile(weights, 95)),
        "extra_sd_pts": float(np.std(extra) * 1000),
        "obs_sd_pts": float(np.std(y) * 1000),
    }


def calibration_table(table: pd.DataFrame, col: str = "pred_full", bins: int = 5) -> pd.DataFrame:
    """Average predicted vs. observed home-road gap in equal-sized bins of the prediction (wOBA points)."""
    t = table.copy()
    t["bin"] = pd.qcut(t[col], bins, labels=False, duplicates="drop")
    rows = []
    for b, d in t.groupby("bin"):
        rows.append({
            "bin": int(b), "pitcher_seasons": len(d),
            "predicted_pts": np.average(d[col], weights=d["weight"]) * 1000,
            "observed_pts": np.average(d["obs"], weights=d["weight"]) * 1000,
        })
    return pd.DataFrame(rows)
