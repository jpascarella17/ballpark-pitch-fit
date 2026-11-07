"""Score pitch archetypes and pitchers by park, from the batted balls they produce.

Park effects (src/park_effects.py) are measured by batter side and batted-ball type.
A pitcher (or an archetype) has a mix of those batted balls. Weighting the park's
effects by that mix gives the expected park effect for that pitcher or archetype.

  edge_pts  expected wOBA points on contact vs. a league-average park (positive helps the pitcher)
  fit_pts   edge_pts minus the same pitcher's (or archetype's) average across all parks, so it says
            how much better this park suits him than the average park does; rank targets with this
  coverage  share of the mix that had a park effect to weight (should be ~1.0)
"""
import pandas as pd


def bip_mix(contact: pd.DataFrame, by: list) -> pd.DataFrame:
    """Share of balls in play in each (batter side, batted-ball type) cell, for each group in `by`."""
    n = contact.groupby(by + ["stand", "bb_type"]).size().rename("bip").reset_index()
    n["share"] = n["bip"] / n.groupby(by)["bip"].transform("sum")
    return n


def expected_edge(mix: pd.DataFrame, effects: pd.DataFrame, by: list) -> pd.DataFrame:
    """Weight each park's batted-ball effects by the mix; returns edge_pts for each group and park."""
    m = mix.merge(effects[["park", "stand", "bb_type", "effect"]], on=["stand", "bb_type"])
    m["w"] = m["share"] * m["effect"]
    out = m.groupby(by + ["park"]).agg(w=("w", "sum"), coverage=("share", "sum")).reset_index()
    out["edge_pts"] = -out["w"] / out["coverage"] * 1000
    return out.drop(columns="w")


def archetype_park_edges(contact: pd.DataFrame, effects: pd.DataFrame, min_bip: int = 2000) -> pd.DataFrame:
    """Expected park effect for each pitch archetype (and pitcher hand) at each park."""
    by = ["archetype", "p_throws"]
    mix = bip_mix(contact, by)
    mix = mix[mix.groupby(by)["bip"].transform("sum") >= min_bip]
    out = expected_edge(mix, effects, by)
    out["fit_pts"] = out["edge_pts"] - out.groupby(by)["edge_pts"].transform("mean")
    return out


def best_archetypes_for_park(arch: pd.DataFrame, park: str, hand: str = None, n: int = 5) -> pd.DataFrame:
    """Archetypes that suit a park best (largest fit_pts)."""
    a = arch[arch["park"] == park]
    if hand is not None:
        a = a[a["p_throws"] == hand]
    return a.sort_values("fit_pts", ascending=False).head(n)[["archetype", "p_throws", "fit_pts", "edge_pts"]]


def archetype_batted_ball_profile(contact: pd.DataFrame, min_bip: int = 2000) -> pd.DataFrame:
    """Batted-ball type shares for each archetype and pitcher hand (for slide text)."""
    by = ["archetype", "p_throws"]
    n = contact.groupby(by + ["bb_type"]).size().rename("bip").reset_index()
    n["share"] = n["bip"] / n.groupby(by)["bip"].transform("sum")
    n = n[n.groupby(by)["bip"].transform("sum") >= min_bip]
    return n.pivot_table(index=by, columns="bb_type", values="share").round(3).reset_index()


def pitcher_fit(contact: pd.DataFrame, effects: pd.DataFrame, min_bip: int = 300) -> pd.DataFrame:
    """Expected park effect for each pitcher at each park, from his own batted-ball mix."""
    mix = bip_mix(contact, ["pitcher"])
    mix = mix[mix.groupby("pitcher")["bip"].transform("sum") >= min_bip]
    out = expected_edge(mix, effects, ["pitcher"])
    out["fit_pts"] = out["edge_pts"] - out.groupby("pitcher")["edge_pts"].transform("mean")
    return out


def best_pitchers_for_park(fit: pd.DataFrame, park: str, names: pd.Series, n: int = 10) -> pd.DataFrame:
    """Pitchers whose batted-ball mix fits a park best."""
    f = fit[fit["park"] == park].copy()
    f["name"] = f["pitcher"].map(names)
    return f.sort_values("fit_pts", ascending=False).head(n)[["name", "fit_pts", "edge_pts"]]
