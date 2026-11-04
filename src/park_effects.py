"""Park effects by pitch archetype, measured on balls in play.

Idea: expected wOBA (from exit velocity and launch angle) does not know which park
the ball was hit in. The gap between actual and expected wOBA on contact therefore
picks up the park (plus defense, weather, and luck). Both home and visiting pitchers
throw in every park, so team quality mostly cancels out.
"""
import pandas as pd


def contact_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Balls in play that count as plate appearances, with actual and expected wOBA."""
    c = df[(df["type"] == "X") & (df["woba_denom"] == 1)].copy()
    c = c.dropna(subset=["woba_value", "estimated_woba_using_speedangle", "archetype", "park"])
    c["woba_gap"] = c["woba_value"] - c["estimated_woba_using_speedangle"]
    c["is_hr"] = (c["events"] == "home_run").astype(int)
    c["is_gb"] = (c["bb_type"] == "ground_ball").astype(int)
    return c


def park_archetype_effects(contact: pd.DataFrame, by_hand: bool = True, shrink_k: int = 200) -> pd.DataFrame:
    """Effect of each park on each archetype, relative to the league average for that archetype.

    effect        shrunk (actual - expected) wOBA on contact vs. league; positive helps hitters
    pitcher_edge  -effect; positive means the park helps this archetype
    shrink_k      balls in play needed before an effect is trusted about 50%;
                  small samples get pulled toward zero
    """
    lg_keys = ["archetype"] + (["p_throws"] if by_hand else [])
    park_keys = ["park"] + lg_keys

    league = contact.groupby(lg_keys).agg(
        lg_gap=("woba_gap", "mean"),
        lg_hr_rate=("is_hr", "mean"),
        lg_gb_rate=("is_gb", "mean"),
    ).reset_index()

    park = contact.groupby(park_keys).agg(
        bip=("woba_gap", "size"),
        gap=("woba_gap", "mean"),
        hr_rate=("is_hr", "mean"),
        gb_rate=("is_gb", "mean"),
    ).reset_index()

    out = park.merge(league, on=lg_keys)
    out["effect_raw"] = out["gap"] - out["lg_gap"]
    out["effect"] = out["effect_raw"] * out["bip"] / (out["bip"] + shrink_k)
    out["pitcher_edge"] = -out["effect"]
    out["hr_diff"] = out["hr_rate"] - out["lg_hr_rate"]
    out["gb_diff"] = out["gb_rate"] - out["lg_gb_rate"]
    return out


def best_archetypes(effects: pd.DataFrame, park: str, hand: str = None, n: int = 5, min_bip: int = 100) -> pd.DataFrame:
    """Archetypes that do best at a park (largest pitcher_edge)."""
    e = effects[(effects["park"] == park) & (effects["bip"] >= min_bip)]
    if hand is not None and "p_throws" in e.columns:
        e = e[e["p_throws"] == hand]
    return e.sort_values("pitcher_edge", ascending=False).head(n)
