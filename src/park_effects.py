"""Park effects, measured on balls in play.

Idea: expected wOBA (built from exit velocity and launch angle) does not know which park
the ball was hit in. The gap between actual and expected wOBA on contact therefore picks
up the park (plus defense, weather, and luck). Both home and visiting pitchers throw in
every park, so team quality mostly cancels out.

Park effects are measured by batter side and batted-ball type (ground ball, line drive,
fly ball, popup). Effects measured per pitch archetype turned out to be mostly noise in the
split-half check (see notebook 03), so archetypes and pitchers are scored afterwards, from
the batted balls they produce (see src/fit.py).
"""
import pandas as pd

BB_TYPES = ["ground_ball", "line_drive", "fly_ball", "popup"]


def contact_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Balls in play that count as plate appearances, with actual and expected wOBA."""
    c = df[(df["type"] == "X") & (df["woba_denom"] == 1)].copy()
    c = c.dropna(subset=["woba_value", "estimated_woba_using_speedangle", "archetype", "park", "stand", "bb_type"])
    c = c[c["bb_type"].isin(BB_TYPES)]
    c["woba_gap"] = c["woba_value"] - c["estimated_woba_using_speedangle"]
    c["is_hr"] = (c["events"] == "home_run").astype(int)
    return c


def park_bb_effects(contact: pd.DataFrame, shrink_k: int = 200) -> pd.DataFrame:
    """Effect of each park on each (batter side, batted-ball type), relative to the league.

    effect        shrunk (actual - expected) wOBA on contact vs. league average for the same
                  batter side and batted-ball type; positive helps hitters
    pitcher_edge  -effect; positive means the park helps pitchers on this kind of contact
    hr_diff       home runs per ball in play vs. league, same batter side and batted-ball type
    shrink_k      balls in play needed before an effect is trusted about 50%
    """
    keys = ["stand", "bb_type"]
    league = contact.groupby(keys).agg(lg_gap=("woba_gap", "mean"), lg_hr=("is_hr", "mean")).reset_index()
    park = contact.groupby(["park"] + keys).agg(
        bip=("woba_gap", "size"), gap=("woba_gap", "mean"), hr=("is_hr", "mean")
    ).reset_index()
    out = park.merge(league, on=keys)
    out["effect_raw"] = out["gap"] - out["lg_gap"]
    out["effect"] = out["effect_raw"] * out["bip"] / (out["bip"] + shrink_k)
    out["pitcher_edge"] = -out["effect"]
    out["hr_diff"] = out["hr"] - out["lg_hr"]
    return out


def park_archetype_effects(contact: pd.DataFrame, by_hand: bool = True, shrink_k: int = 200) -> pd.DataFrame:
    """Park effect per pitch archetype (the first approach, kept to document the stability test).

    Split-half correlation of the archetype-specific part was about 0.05, so this is not
    used for the final results.
    """
    lg_keys = ["archetype"] + (["p_throws"] if by_hand else [])
    league = contact.groupby(lg_keys).agg(lg_gap=("woba_gap", "mean")).reset_index()
    park = contact.groupby(["park"] + lg_keys).agg(bip=("woba_gap", "size"), gap=("woba_gap", "mean")).reset_index()
    out = park.merge(league, on=lg_keys)
    out["effect_raw"] = out["gap"] - out["lg_gap"]
    out["effect"] = out["effect_raw"] * out["bip"] / (out["bip"] + shrink_k)
    return out


def _specific_part(e: pd.DataFrame) -> pd.Series:
    """Effect with each park's overall (ball-in-play weighted) effect removed."""
    wsum = (e["effect_raw"] * e["bip"]).groupby(e["park"]).transform("sum")
    return e["effect_raw"] - wsum / e["bip"].groupby(e["park"]).transform("sum")


def split_half_stability(contact: pd.DataFrame, level: str = "batted_ball", min_bip: int = 100) -> dict:
    """Do park effects from the earlier seasons predict the later seasons?

    level: "batted_ball" (park x batter side x batted-ball type) or "archetype"
           (park x pitch archetype x pitcher hand)
    Returns the number of cells, the overall correlation, and the correlation of the
    specific part only (each park's overall effect removed).
    """
    if level == "batted_ball":
        fn, keys = park_bb_effects, ["park", "stand", "bb_type"]
    elif level == "archetype":
        fn, keys = park_archetype_effects, ["park", "archetype", "p_throws"]
    else:
        raise ValueError("level must be 'batted_ball' or 'archetype'")

    seasons = sorted(contact["game_year"].unique())
    half = len(seasons) // 2
    a = fn(contact[contact["game_year"].isin(seasons[:half])])
    b = fn(contact[contact["game_year"].isin(seasons[half:])])
    a["specific"] = _specific_part(a)
    b["specific"] = _specific_part(b)
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    m = m[(m["bip_a"] >= min_bip) & (m["bip_b"] >= min_bip)]
    return {
        "cells": len(m),
        "overall": float(m["effect_raw_a"].corr(m["effect_raw_b"])),
        "specific": float(m["specific_a"].corr(m["specific_b"])),
    }
