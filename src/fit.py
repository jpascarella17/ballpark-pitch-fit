"""Score how well each pitcher's pitch mix fits each park."""
import pandas as pd


def pitcher_bip_mix(contact: pd.DataFrame, min_bip: int = 150) -> pd.DataFrame:
    """Share of each pitcher's balls in play by archetype.

    Park effects act on balls in play, so the mix is weighted by balls in play
    rather than by raw pitch counts.
    """
    mix = contact.groupby(["pitcher", "p_throws", "archetype"]).size().rename("bip").reset_index()
    total = mix.groupby("pitcher")["bip"].transform("sum")
    mix = mix[total >= min_bip].copy()
    mix["share"] = mix["bip"] / mix.groupby("pitcher")["bip"].transform("sum")
    return mix


def park_fit(mix: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """Expected park effect for each pitcher at each park.

    park_effect_pts  wOBA points on contact vs. league average (negative helps the pitcher)
    fit_score        how much better this park suits the pitcher than the average park does
                     (positive = better than average fit); use this to rank targets
    coverage         share of the pitcher's mix that had enough data at that park
    """
    m = mix.merge(effects[["park", "archetype", "p_throws", "effect"]], on=["archetype", "p_throws"])
    m["weighted"] = m["share"] * m["effect"]
    grouped = m.groupby(["pitcher", "park"]).agg(weighted=("weighted", "sum"), coverage=("share", "sum")).reset_index()
    grouped["park_effect_pts"] = grouped["weighted"] / grouped["coverage"] * 1000
    avg = grouped.groupby("pitcher")["park_effect_pts"].transform("mean")
    grouped["fit_score"] = -(grouped["park_effect_pts"] - avg)
    return grouped.drop(columns="weighted")


def best_pitchers_for_park(fit: pd.DataFrame, park: str, names: pd.Series, n: int = 10, min_coverage: float = 0.8) -> pd.DataFrame:
    """Pitchers whose mix fits a park best."""
    f = fit[(fit["park"] == park) & (fit["coverage"] >= min_coverage)].copy()
    f["name"] = f["pitcher"].map(names)
    return f.sort_values("fit_score", ascending=False).head(n)[["name", "fit_score", "park_effect_pts", "coverage"]]
