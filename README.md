# Ballpark Pitch Fit

Which pitch shapes thrive (or struggle) in each MLB ballpark, and which pitchers are the best fit for each park?

The goal is to give a front office a data-driven way to answer *"Will this pitcher's stuff play in our park?"* when evaluating free agents and trade targets.

## Approach

1. **Data.** Pitch-level Statcast data for the regular seasons 2022-2026 (about 3.5 million pitches), pulled with `pybaseball`.
2. **Pitch archetypes.** Each pitch is grouped by pitch type, arm slot (Statcast arm angle, 2 tiers), and induced vertical break (2 tiers), giving 32 archetypes. Velocity stays in the data and on the slides but does not split the groups, because park effects act on batted balls.
3. **Park effects on contact.** On balls in play, compare actual wOBA to expected wOBA (built from exit velocity and launch angle, which does not know the park). The gap, relative to the league average for the same kind of contact and shrunk toward zero when samples are small, is the park effect. It is measured by **batter side and batted-ball type** (ground ball, line drive, fly ball, popup). Home and visiting pitchers both throw in every park, so team quality mostly cancels out.
4. **Archetype and pitcher fit.** Each archetype or pitcher produces a mix of those batted balls. Weighting each park's effects by that mix gives the expected park effect. The fit score removes the park's overall effect, so it shows how much better a park suits him than it suits a typical pitcher (otherwise every pitcher would just inherit the park factor).
5. **Backtest.** Check whether fit scores predict what happened to pitchers who changed parks, against a plain park-factor baseline.
6. **Slides.** One slide per park plus summary slides on the biggest mismatches and how a team would use this.

## Status

- [x] **Setup:** repo, environment, folder structure
- [x] **01 Data pull:** 5 seasons downloaded and checked
- [x] **02 Archetypes:** 32 archetypes, cut points saved
- [ ] **03 Park effects:** effects by batter side and batted-ball type, archetype fit scores, charts
- [ ] **04 Fit scores:** pitcher fit scores and free-agent targets
- [ ] **Backtest:** pitchers who changed parks
- [ ] **Slides:** 30 park slides, summary, limitations, usage

## Key findings so far

**Park effects are real, but they do not vary by pitch archetype the way I first assumed.** I split the seasons into an early half and a late half and checked whether park effects from one half predict the other:

| Level of detail | Cells | Correlation of the park-specific part |
|---|---|---|
| Park x pitch archetype x pitcher hand | 782 | 0.05 |
| Park x batter side x batted-ball type | 240 | 0.50 |

Park-wide effects repeat across halves (correlation 0.59 across parks). Effects measured for each pitch archetype directly did not (0.05, indistinguishable from noise), so the project now measures park effects on batted-ball types and scores pitches and pitchers by the batted balls they produce.

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

In VS Code, select the `.venv` interpreter/kernel, then run the notebooks in order starting with `notebooks/01_pull_data.ipynb`. Each season is cached in `data/raw/` after the first download.

## Limitations

- Weather, defense, umpires, and lineup quality are not modeled; some leak into the actual-minus-expected gap.
- Park effects are measured by batter side and batted-ball type, not by spray direction or individual pitch type.
- Sutter Health Park has two seasons of data and George M. Steinbrenner Field one, so those parks are low-confidence.
- Fit scores are estimates from past data, not predictions of exact performance.
