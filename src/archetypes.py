"""Group pitches into archetypes by type, arm slot, velocity, and vertical shape."""
import numpy as np
import pandas as pd

# Statcast pitch_type -> pitch group. Rare types (eephus, pitchouts, etc.) are dropped.
PITCH_GROUPS = {
    "FF": "Four-seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Sweeper",  # slurves are lumped in with sweepers to keep samples large enough
    "CU": "Curveball",
    "KC": "Curveball",
    "CH": "Changeup",
    "FS": "Splitter",
}

# Each pitch group is split into (slot tiers x velocity tiers x IVB tiers) archetypes.
# More tiers = more specific archetypes but fewer balls in play per park/archetype cell.
# 2 x 2 x 2 = 8 archetypes per pitch group is a safe starting point for ~5 seasons of data.
# Try 3 slot tiers (["low", "mid", "high"]) only after checking cell sizes in notebook 03.
SLOT_LABELS = ["lower", "higher"]
VELO_LABELS = ["slower", "faster"]
IVB_LABELS = ["low-IVB", "high-IVB"]


def add_shape_features(df: pd.DataFrame, slot_source: str = "auto") -> pd.DataFrame:
    """Add pitch group, induced vertical break, arm-side run, and an arm-slot value.

    slot_source:
      "auto"           use Statcast arm_angle if it is >=90% populated, else release height
      "arm_angle"      use arm_angle and drop pitches where it is missing
      "release_height" use release_pos_z
    """
    df = df[df["pitch_type"].isin(PITCH_GROUPS)].copy()
    df["pitch_group"] = df["pitch_type"].map(PITCH_GROUPS)

    # pfx_z is vertical movement in feet; pfx_x is horizontal movement in feet from the
    # catcher's view. For RHP, arm-side run is negative in that view, so flip the sign
    # for righties to make arm-side run positive for both hands.
    df["ivb_in"] = df["pfx_z"] * 12
    sign = np.where(df["p_throws"] == "R", -1, 1)
    df["arm_side_in"] = df["pfx_x"] * 12 * sign

    has_angle = "arm_angle" in df.columns
    use_angle = slot_source == "arm_angle" or (
        slot_source == "auto" and has_angle and df["arm_angle"].notna().mean() >= 0.9
    )
    if use_angle:
        df["slot_value"] = df["arm_angle"]
        print("Arm slot source: Statcast arm_angle")
    else:
        df["slot_value"] = df["release_pos_z"]
        print("Arm slot source: release height (arm_angle missing or too sparse)")
    return df


def _tier(df: pd.DataFrame, col: str, labels: list) -> pd.Series:
    """Percentile tier of `col` within each pitch group (e.g. low/mid/high)."""
    n = len(labels)
    pct = df.groupby("pitch_group")[col].rank(pct=True, method="first")
    idx = np.ceil(pct * n).clip(1, n) - 1
    out = pd.Series(np.nan, index=df.index, dtype="object")
    ok = idx.notna()
    out[ok] = np.asarray(labels, dtype=object)[idx[ok].astype(int).to_numpy()]
    return out


def add_archetypes(df, slot_labels=SLOT_LABELS, velo_labels=VELO_LABELS, ivb_labels=IVB_LABELS):
    """Add tier columns and an `archetype` label like 'Four-seam | higher slot | faster | high-IVB'."""
    df = df.copy()
    df["slot_tier"] = _tier(df, "slot_value", slot_labels)
    df["velo_tier"] = _tier(df, "release_speed", velo_labels)
    df["ivb_tier"] = _tier(df, "ivb_in", ivb_labels)
    ok = df[["slot_tier", "velo_tier", "ivb_tier"]].notna().all(axis=1)
    df["archetype"] = np.nan
    df["archetype"] = df["archetype"].astype("object")
    df.loc[ok, "archetype"] = (
        df.loc[ok, "pitch_group"] + " | " + df.loc[ok, "slot_tier"] + " slot | "
        + df.loc[ok, "ivb_tier"]
    )
    return df
    

