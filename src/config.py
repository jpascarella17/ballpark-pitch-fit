"""Project-wide settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"            # downloaded Statcast files (git-ignored)
INTERIM = ROOT / "data" / "interim"    # large intermediate files (git-ignored)
PROCESSED = ROOT / "data" / "processed"  # small summary tables (committed)
FIGURES = ROOT / "slides" / "figures"  # charts used on the slides

# Seasons to analyze. Change this to widen or narrow the window.
SEASONS = list(range(2022, 2027))

# Regular-season pull window (month-day). Postseason is left out.
SEASON_START = "03-15"
SEASON_END = "10-10"
