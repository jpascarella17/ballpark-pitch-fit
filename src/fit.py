"""Score pitch archetypes and pitchers by park, from the batted balls they produce.

Park effects (src/park_effects.py) are measured by batter side and batted-ball type.
A pitcher (or an archetype) has a mix of those batted balls. Weighting the park's
effects by that mix gives the expected park effect for that pitcher or archetype.

  edge_pts  expected wOBA points on contact vs. a league-average park (positive helps the pitcher)
  fit_pts   edge_pts with two things removed: the pitcher's (or archetype's) average across all parks,
            and the park's average across all pitchers (or archetypes). What is left is how much
            better this park suits him than it suits a typical pitcher; rank "fit" with this.
            It is RELATIVE: at a very hitter-friendly park every pitcher is hurt, and the best fit
            is just the pitcher who is hurt least.
  proj_woba projected wOBA allowed at the park: his park-neutral wOBA allowed (talent_woba, shrunk
            toward the league average) adjusted by edge_pts. Lower is better; this one combines
            talent and park, so use it to say who would actually pitch best there.
  coverage  share of the mix that had a park effect to weight (should be ~1.0)
"""
import pandas as pd

# Parks that are no longer (or were only temporarily) a team's home. They stay in the park-effect
# data (they feed the league baselines and visiting pitchers' results) but are left out of
# archetype and pitcher scoring and the per-park slides.
INACTIVE_PARKS = {"Oakland Coliseum", "George M. Steinbrenner Field"}


def _interaction(out: pd.DataFrame, unit_cols: list, group_cols: list = None) -> pd.Series:
    """edge_pts minus each unit's average across parks and each park's average across units.

    group_cols limits the "typical unit" to a subgroup (for example, the same pitcher hand).
    """
    group_cols = group_cols or []
    unit_mean = out.groupby(unit_cols)["edge_pts"].transform("mean")
    park_mean = out.groupby(["park"] + group_cols)["edge_pts"].transform("mean")
    grand = out.groupby(group_cols)["edge_pts"].transform("mean") if group_cols else out["edge_pts"].mean()
    return out["edge_pts"] - unit_mean - park_mean + grand


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


def archetype_park_edges(contact: pd.DataFrame, effects: pd.DataFrame, min_bip: int = 2000,
                         exclude_parks=INACTIVE_PARKS) -> pd.DataFrame:
    """Expected park effect for each pitch archetype (and pitcher hand) at each park."""
    effects = effects[~effects["park"].isin(exclude_parks)]
    by = ["archetype", "p_throws"]
    mix = bip_mix(contact, by)
    mix = mix[mix.groupby(by)["bip"].transform("sum") >= min_bip]
    out = expected_edge(mix, effects, by)
    out["fit_pts"] = _interaction(out, by, group_cols=["p_throws"])
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


def pitcher_fit(contact: pd.DataFrame, effects: pd.DataFrame, min_bip: int = 300,
                exclude_parks=INACTIVE_PARKS) -> pd.DataFrame:
    """Expected park effect for each pitcher at each park, from his own batted-ball mix."""
    effects = effects[~effects["park"].isin(exclude_parks)]
    mix = bip_mix(contact, ["pitcher"])
    mix = mix[mix.groupby("pitcher")["bip"].transform("sum") >= min_bip]
    out = expected_edge(mix, effects, ["pitcher"])
    out["fit_pts"] = _interaction(out, ["pitcher"])
    return out


def pitcher_talent(df: pd.DataFrame, effects: pd.DataFrame, recent_seasons: int = 2,
                   shrink_pa: int = 500, min_pa: int = 100) -> pd.DataFrame:
    """Park-neutral wOBA allowed for each pitcher over his most recent seasons.

    Each ball in play is adjusted by the park effect for its park, batter side, and batted-ball
    type, so a pitcher who worked in a hitter's park is not penalized (and vice versa). The result
    is then shrunk toward the league average, because wOBA allowed is noisy: shrink_pa is the
    number of plate appearances of league-average results blended in (a judgment call, not fitted).

    `effects` should include every park (including inactive ones), since pitchers worked in all of them.
    Returns one row per pitcher (index = pitcher id) with pa, talent_woba, bip_rate.
    """
    last = sorted(df["game_year"].unique())[-recent_seasons:]
    pa = df[df["game_year"].isin(last) & (df["woba_denom"] == 1)].copy()
    pa = pa.merge(effects[["park", "stand", "bb_type", "effect"]], on=["park", "stand", "bb_type"], how="left")
    pa["effect"] = pa["effect"].fillna(0.0)          # walks, strikeouts, and unmatched contact: no park effect
    pa["woba_value"] = pa["woba_value"].fillna(0.0)
    pa["is_bip"] = (pa["type"] == "X").astype(int)

    g = pa.groupby("pitcher").agg(
        pa=("woba_value", "size"), woba_sum=("woba_value", "sum"),
        effect_sum=("effect", "sum"), bip=("is_bip", "sum"),
    )
    g = g[g["pa"] >= min_pa].copy()
    g["neutral_woba"] = (g["woba_sum"] - g["effect_sum"]) / g["pa"]
    league = (g["neutral_woba"] * g["pa"]).sum() / g["pa"].sum()
    g["talent_woba"] = (g["neutral_woba"] * g["pa"] + league * shrink_pa) / (g["pa"] + shrink_pa)
    g["bip_rate"] = g["bip"] / g["pa"]
    return g[["pa", "neutral_woba", "talent_woba", "bip_rate"]]


def project_park_woba(fit: pd.DataFrame, talent: pd.DataFrame) -> pd.DataFrame:
    """Projected wOBA allowed for each pitcher at each park: talent adjusted by the park's edge.

    edge_pts is per ball in play, so it is scaled by the share of his plate appearances that end
    in a ball in play. proj_rank is 1 for the lowest projected wOBA allowed at that park.
    """
    p = fit.merge(talent, left_on="pitcher", right_index=True)
    p["proj_woba"] = p["talent_woba"] - p["edge_pts"] / 1000 * p["bip_rate"]
    p["proj_rank"] = p.groupby("park")["proj_woba"].rank(method="min")
    return p


def best_pitchers_for_park(fit: pd.DataFrame, park: str, names: pd.Series, n: int = 10) -> pd.DataFrame:
    """Pitchers whose batted-ball mix fits a park best."""
    f = fit[fit["park"] == park].copy()
    f["name"] = f["pitcher"].map(names)
    return f.sort_values("fit_pts", ascending=False).head(n)[["name", "fit_pts", "edge_pts"]]
