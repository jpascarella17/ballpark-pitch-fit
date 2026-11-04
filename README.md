# Ballpark Pitch Fit

Which pitch shapes thrive (or struggle) in each MLB ballpark, and which pitchers are the best fit for each park?

The goal is to give a front office a data-driven way to answer *"Will this pitcher's stuff play in our park?"* when evaluating free agents and trade targets.

## Approach

1. **Data.** Pull pitch-level Statcast data (regular seasons 2022-2026) with `pybaseball`.
2. **Pitch archetypes.** Group pitches by type, arm slot, velocity tier, and induced vertical break tier (e.g. `Four-seam | higher slot | faster | high-IVB`). Grouping by archetype avoids the noise of individual pitcher home/road splits.
3. **Park effects by archetype.** On balls in play, compare actual wOBA to expected wOBA (built from exit velocity and launch angle, which doesn't know the park). The gap, relative to the league average for that same archetype and shrunk toward zero when samples are small, is the park effect. Home and visiting pitchers both throw in every park, so team quality mostly cancels out.
4. **Pitcher fit scores.** Weight each pitcher's mix of archetypes (by balls in play) by each park's effects. The fit score is how much better a park suits him than the average park does.
5. **Backtest.** Check whether the fit scores predict what happened to pitchers who changed parks, against a plain park-factor baseline.
6. **Slides.** One slide per park plus summary slides on the biggest mismatches and how a team would use this.

## Status

Target weeks are counted from the start of the project. Fill in dates as you go.

- [ ] **Setup:** repo, environment, folder structure (week 1)
- [ ] **01 Data pull:** Statcast downloaded and checked (weeks 1-2)
- [ ] **02 Archetypes:** pitch groups defined and charted (weeks 2-3)
- [ ] **03 Park effects:** effects by park, archetype, and handedness; split-half stability check (weeks 3-5)
- [ ] **04 Fit scores:** pitcher fit scores and free-agent targets (weeks 5-6)
- [ ] **Backtest:** pitchers who changed parks (weeks 6-7)
- [ ] **Slides:** 30 park slides, summary, limitations, usage (weeks 7-9)

## Key findings so far

_Add your best chart and 2-3 bullet findings here as results come in._

## Repo layout

```
src/                  reusable code (data pull, archetypes, park effects, fit scores, plots)
notebooks/            01_pull_data, 02_archetypes, 03_park_effects, 04_fa_targets
data/raw/             downloaded Statcast files (git-ignored)
data/interim/         large intermediate files (git-ignored)
data/processed/       small summary tables (committed)
slides/               final deck and slides/figures/ for charts
```

## Setup

```bash
python -m venv .venv
# Mac/Linux:  source .venv/bin/activate
# Windows:    .venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter/kernel, then run the notebooks in order starting with `notebooks/01_pull_data.ipynb`. The first data pull takes a while; each season is cached in `data/raw/`.

## Limitations

- Weather, defense, umpires, and lineup quality are not modeled; some leak into the actual-minus-expected gap.
- Archetypes capture shape, velocity, and slot, not command, deception, or sequencing.
- Parks that changed or were temporarily replaced (Athletics, Rays) and neutral-site games need care; see `src/data.py`.
- Small samples are shrunk toward zero, but thin archetype/park cells should still be read with caution.
- Fit scores are estimates from past data, not predictions of exact performance.
