"""Pull, load, and label Statcast pitch-level data."""
import pandas as pd

from .config import RAW, SEASONS, SEASON_START, SEASON_END

# Columns kept from the raw pull (anything missing from Statcast is skipped).
KEEP_COLS = [
    "game_pk", "game_date", "game_year", "game_type", "home_team", "away_team",
    "inning_topbot", "at_bat_number", "pitch_number",
    "pitcher", "player_name", "p_throws", "batter", "stand",
    "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
    "plate_x", "plate_z",
    "type", "events", "description", "bb_type", "balls", "strikes",
    "launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
]

# Smaller column set for the analysis notebooks (keeps memory use down).
ANALYSIS_COLS = [
    "game_pk", "game_year", "home_team", "pitcher", "player_name", "p_throws", "stand",
    "pitch_type", "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
    "type", "events", "bb_type", "launch_speed", "launch_angle",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
]

# Statcast home_team code -> park name. Names are kept consistent across seasons
# (naming-rights changes are ignored) so each park aggregates as one unit.
PARKS = {
    "ARI": "Chase Field", "AZ": "Chase Field",
    "ATL": "Truist Park",
    "BAL": "Oriole Park at Camden Yards",
    "BOS": "Fenway Park",
    "CHC": "Wrigley Field",
    "CWS": "Rate Field",
    "CIN": "Great American Ball Park",
    "CLE": "Progressive Field",
    "COL": "Coors Field",
    "DET": "Comerica Park",
    "HOU": "Daikin Park",
    "KC": "Kauffman Stadium",
    "LAA": "Angel Stadium",
    "LAD": "Dodger Stadium",
    "MIA": "loanDepot park",
    "MIL": "American Family Field",
    "MIN": "Target Field",
    "NYM": "Citi Field",
    "NYY": "Yankee Stadium",
    "PHI": "Citizens Bank Park",
    "PIT": "PNC Park",
    "SD": "Petco Park",
    "SEA": "T-Mobile Park",
    "SF": "Oracle Park",
    "STL": "Busch Stadium",
    "TB": "Tropicana Field",
    "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",
    "WSH": "Nationals Park",
    "OAK": "Oakland Coliseum",
    "ATH": "Sutter Health Park",
}

# Teams that played home games somewhere other than their usual park.
# Keys are (home_team code, season). VERIFY these against the seasons you use.
PARK_OVERRIDES = {
    ("TB", 2025): "George M. Steinbrenner Field",
    ("OAK", 2025): "Sutter Health Park",
    ("OAK", 2026): "Sutter Health Park",
}

# Optional: game_pk values for neutral-site games (international series, etc.)
# to drop from the analysis. Look the IDs up and add them here.
NEUTRAL_SITE_GAME_PKS: set = set()


def pull_season(year: int, force: bool = False):
    """Download one regular season of Statcast data to data/raw/ as parquet.

    Skips the download if the file already exists (use force=True to redo it).
    """
    from pybaseball import statcast, cache  # imported here so other code works without it

    out = RAW / f"statcast_{year}.parquet"
    if out.exists() and not force:
        print(f"{year}: already downloaded, skipping")
        return out

    RAW.mkdir(parents=True, exist_ok=True)
    cache.enable()
    df = statcast(start_dt=f"{year}-{SEASON_START}", end_dt=f"{year}-{SEASON_END}", verbose=True)
    df = df[df["game_type"] == "R"]
    cols = [c for c in KEEP_COLS if c in df.columns]
    df[cols].to_parquet(out, index=False)
    print(f"{year}: saved {len(df):,} pitches to {out.name}")
    return out


def load_seasons(seasons=SEASONS, columns=None) -> pd.DataFrame:
    """Load downloaded seasons into one DataFrame (optionally only some columns)."""
    import pyarrow.parquet as pq

    frames = []
    for year in seasons:
        path = RAW / f"statcast_{year}.parquet"
        if not path.exists():
            print(f"{year}: no file found, skipping (run pull_season first)")
            continue
        cols = None
        if columns is not None:
            have = set(pq.read_schema(path).names)
            cols = [c for c in columns if c in have]
        frames.append(pd.read_parquet(path, columns=cols))
    if not frames:
        raise FileNotFoundError("No Statcast files found in data/raw/. Run notebook 01 first.")
    return pd.concat(frames, ignore_index=True)


def add_park(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `park` column from home_team and season, and drop neutral-site games."""
    df = df.copy()
    df["park"] = df["home_team"].map(PARKS)
    for (team, year), name in PARK_OVERRIDES.items():
        mask = (df["home_team"] == team) & (df["game_year"] == year)
        df.loc[mask, "park"] = name
    athletics = df["home_team"].isin(["OAK", "ATH"])
    df.loc[athletics & (df["game_year"] <= 2024), "park"] = "Oakland Coliseum"
    df.loc[athletics & (df["game_year"] >= 2025), "park"] = "Sutter Health Park"
    unmapped = df.loc[df["park"].isna(), "home_team"].unique()
    if len(unmapped):
        print("WARNING: home_team codes with no park mapping:", list(unmapped))
    if NEUTRAL_SITE_GAME_PKS:
        df = df[~df["game_pk"].isin(NEUTRAL_SITE_GAME_PKS)]
    return df
