"""Build the PowerPoint deck from the saved tables in data/processed/.

    from src.slides import build_deck
    build_deck(author="Your Name")

Slides: title, approach, validation, park-effects overview and heat map, buy-low free agents,
one slide per active park, limitations. Nothing here recomputes the analysis; it only reads the
CSVs the notebooks save, so re-run the notebooks first when the data changes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from .config import FIGURES, PROCESSED, ROOT
from .plots import plot_park_archetypes_both, plot_park_bb

NAVY = RGBColor(0x1B, 0x2A, 0x41)
BLUE = RGBColor(0x1F, 0x61, 0x8D)
TEXT = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x66, 0x66, 0x66)
WARN = RGBColor(0xB9, 0x77, 0x0E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

BB_NAMES = {"ground_ball": "ground balls", "line_drive": "line drives", "fly_ball": "fly balls", "popup": "popups"}
SIDE_NAMES = {"L": "left-handed", "R": "right-handed"}
ROLE_NAMES = {"S": "SP", "R": "RP"}
LOW_CONFIDENCE_SEASONS = 3


# ------------------------------------------------------------------ small helpers
def _read(name):
    path = Path(PROCESSED) / name
    return pd.read_csv(path) if path.exists() else None


def _w(v):
    """wOBA as .xxx"""
    return f"{v:.3f}".lstrip("0")


def _first_last(name):
    """'Skubal, Tarik' -> 'Tarik Skubal'"""
    if "," in str(name):
        last, first = [s.strip() for s in str(name).split(",", 1)]
        return f"{first} {last}"
    return str(name)


def _add_text(slide, x, y, w, h, items, size=14, color=TEXT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for k, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = item["text"]
        run.font.size = Pt(item.get("size", size))
        run.font.bold = item.get("bold", False)
        run.font.color.rgb = item.get("color", color)
        p.space_after = Pt(item.get("space", 4))
    return box


def _new_slide(prs, title, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    _add_text(slide, 0.4, 0.17, 10.8, 0.7, [{"text": title, "size": 26, "bold": True, "color": WHITE}])
    if subtitle:
        _add_text(slide, 9.0, 0.3, 4.0, 0.5, [{"text": subtitle, "size": 12, "color": RGBColor(0xF5, 0xB0, 0x41)}])
    return slide


def _add_table(slide, x, y, rows, col_widths, font=11, row_h=0.34):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(col_widths)), Inches(row_h * len(rows)))
    tbl = shape.table
    for j, cw in enumerate(col_widths):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(font)
                    r.font.bold = i == 0
    return tbl


def _park_overall(effects, parks):
    """Ball-in-play weighted effect of each park on pitchers, in wOBA points (positive helps pitchers)."""
    out = {}
    for park in parks:
        e = effects[effects["park"] == park]
        out[park] = float(np.average(e["pitcher_edge"], weights=e["bip"]) * 1000)
    return pd.Series(out).sort_values(ascending=False)


# ------------------------------------------------------------------ figures made for the deck
def _overall_chart(overall, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    s = overall.sort_values()
    ax.barh(s.index, s.values, color=["#b03a2e" if v < 0 else "#1f618d" for v in s.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("wOBA points per ball in play vs. a league-average park (+ helps pitchers)", fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _heatmap(effects, overall, path):
    cols = [(s, t) for s in ["L", "R"] for t in ["ground_ball", "line_drive", "fly_ball", "popup"]]
    order = list(overall.index)
    mat = np.full((len(order), len(cols)), np.nan)
    for i, park in enumerate(order):
        for j, (s, t) in enumerate(cols):
            v = effects[(effects["park"] == park) & (effects["stand"] == s) & (effects["bb_type"] == t)]["pitcher_edge"]
            if len(v):
                mat[i, j] = v.iloc[0] * 1000
    vmax = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{s}\n{BB_NAMES[t].split()[0]}" for s, t in cols], fontsize=7)
    ax.xaxis.tick_top()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(mat[i, j]) > 0.55 * vmax else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ slides
def _title_slide(prs, author):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    _add_text(slide, 0.9, 2.2, 11.5, 1.2, [{"text": "Ballpark Pitch Fit", "size": 48, "bold": True, "color": WHITE}])
    _add_text(slide, 0.9, 3.5, 11.5, 1.2, [
        {"text": "How MLB ballparks change pitcher results, and what that means for free agency",
         "size": 22, "color": RGBColor(0xDD, 0xE6, 0xF0)}])
    _add_text(slide, 0.9, 5.4, 11.5, 1.0, [
        {"text": author, "size": 18, "color": WHITE},
        {"text": "Data: Baseball Savant (Statcast), 2022-2026 regular seasons", "size": 13,
         "color": RGBColor(0xAA, 0xBB, 0xCC)}])


def _approach_slide(prs):
    s = _new_slide(prs, "The question and the approach")
    _add_text(s, 0.6, 1.3, 12.1, 5.6, [
        {"text": "Question: will this pitcher's results hold up in our ballpark?", "size": 22, "bold": True, "color": BLUE, "space": 14},
        {"text": "1.  Measure each park's effect on balls in play, by batter side and batted-ball type.", "size": 18, "space": 10},
        {"text": "Expected wOBA (from exit velocity and launch angle) does not know the park. The gap between actual and "
                 "expected wOBA, compared with the league for the same kind of contact, is the park effect.", "size": 14, "color": MUTED, "space": 14},
        {"text": "2.  Describe pitchers and pitch types by the batted balls they produce.", "size": 18, "space": 10},
        {"text": "A sinker-baller and a high-fastball pitcher give up very different mixes of ground balls and fly balls, "
                 "so one park treats them differently.", "size": 14, "color": MUTED, "space": 14},
        {"text": "3.  Project each pitcher's wOBA allowed at each park.", "size": 18, "space": 10},
        {"text": "His park-neutral wOBA allowed (pulled toward the league average, since results are noisy) adjusted by that "
                 "park's expected effect on him.", "size": 14, "color": MUTED, "space": 14},
        {"text": "4.  Test it.", "size": 18, "space": 10},
        {"text": "Do park effects repeat across seasons, and do they predict real home-vs-road results the model never saw?",
         "size": 14, "color": MUTED},
    ])


def _validation_slide(prs, stability, backtest):
    s = _new_slide(prs, "Does it hold up? Testing the model")
    items = []
    if stability is not None:
        row = {r["level"]: r for _, r in stability.iterrows()}
        if "archetype" in row and "batted_ball" in row:
            a, bb = row["archetype"]["specific"], row["batted_ball"]["specific"]
            items += [
                {"text": "Do park effects repeat from one set of seasons to the next?", "size": 20, "bold": True, "color": BLUE, "space": 8},
                {"text": f"Effects by pitch archetype: correlation {a:.2f}. "
                         + ("That is noise, so my first idea (a park rewards specific pitch types) did not hold."
                            if a < 0.2 else "These repeat reasonably well."), "size": 16, "space": 8},
                {"text": f"Effects by batter side and batted-ball type: correlation {bb:.2f}. "
                         + ("These repeat, so the model is built on them."
                            if bb >= 0.3 else "These barely repeat, so treat the park effects with caution."), "size": 16, "space": 18},
            ]
    gain_known = False
    if backtest is not None:
        b = backtest.iloc[0]
        gain_known = b["r2_gain_lo"] > 0
        park_line = (f"The park effect predicts it (correlation {b['corr_park']:.2f}, even though a single pitcher-season is very noisy)."
                     if b["corr_park"] >= 0.1 else
                     f"The park effect only weakly predicts it (correlation {b['corr_park']:.2f}).")
        if gain_known:
            extra_line = (f"The pitcher-specific part adds a small but measurable gain (R-squared gain {b['r2_gain']:.3f}, "
                          f"90% interval {b['r2_gain_lo']:.3f} to {b['r2_gain_hi']:.3f}).")
        else:
            extra_line = (f"The pitcher-specific part adds nothing we can confirm (R-squared gain {b['r2_gain']:.3f}, 90% interval "
                          f"{b['r2_gain_lo']:.3f} to {b['r2_gain_hi']:.3f}).")
            if "extra_sd_pts" in b and "obs_sd_pts" in b:
                extra_line += (f" It varies only about {b['extra_sd_pts']:.0f} points between pitchers against "
                               f"{b['obs_sd_pts']:.0f} points of noise, so a test this size cannot see it.")
        items += [
            {"text": "Do they predict results the model never saw?", "size": 20, "bold": True, "color": BLUE, "space": 8},
            {"text": f"Backtest on {int(b['n'])} pitcher-seasons: how hitters did at a pitcher's home park vs. on the road, "
                     "predicted from the other seasons only.", "size": 16, "space": 8},
            {"text": park_line, "size": 16, "space": 8},
            {"text": extra_line, "size": 16, "space": 18},
        ]
    items.append({"text": ("Takeaway: parks matter a lot, and the pitcher-specific fit adds a small, measurable edge." if gain_known else
                           "Takeaway: parks matter a lot and that is solid. Which pitcher benefits more is a small, unconfirmed tiebreaker."),
                  "size": 18, "bold": True})
    _add_text(s, 0.6, 1.3, 12.1, 5.7, items)


def _overview_slide(prs, effects, overall, fig_dir, low_conf):
    path = Path(fig_dir) / "overall_effects.png"
    _overall_chart(overall, path)
    s = _new_slide(prs, "How much each park helps or hurts pitchers")
    s.shapes.add_picture(str(path), Inches(0.4), Inches(1.1), width=Inches(7.2))
    top_h = ", ".join(overall.index[-3:][::-1])
    top_p = ", ".join(overall.index[:3])
    items = [
        {"text": "Most hitter-friendly", "size": 15, "bold": True, "color": BLUE},
        {"text": top_h, "size": 14, "space": 12},
        {"text": "Most pitcher-friendly", "size": 15, "bold": True, "color": BLUE},
        {"text": top_p, "size": 14, "space": 12},
        {"text": "Scale: these are per ball in play. Roughly two-thirds of plate appearances end in a ball in play, "
                 "so a 25-point park effect moves a pitcher's overall wOBA allowed by about 17 points.", "size": 13, "color": MUTED, "space": 10},
    ]
    if low_conf:
        items.append({"text": "Low confidence (fewer than 3 seasons of data): " + ", ".join(low_conf) + ".", "size": 12, "color": WARN})
    _add_text(s, 7.9, 1.3, 5.0, 5.6, items)


def _heatmap_slide(prs, effects, overall, fig_dir):
    path = Path(fig_dir) / "heatmap.png"
    _heatmap(effects, overall, path)
    s = _new_slide(prs, "Which kind of contact each park punishes")
    s.shapes.add_picture(str(path), Inches(0.4), Inches(1.1), width=Inches(7.2))
    avg = {}
    for bb, d in effects[effects["park"].isin(overall.index)].groupby("bb_type"):
        avg[bb] = float(np.average(d["pitcher_edge"].abs(), weights=d["bip"]) * 1000)
    ranked = sorted(avg, key=avg.get, reverse=True)
    biggest = ", ".join(f"{BB_NAMES[b]} ({avg[b]:.0f} pts)" for b in ranked[:2])
    smallest = " and ".join(f"{BB_NAMES[b]} ({avg[b]:.0f} pts)" for b in ranked[2:])
    _add_text(s, 7.9, 1.3, 5.0, 5.6, [
        {"text": "How to read it", "size": 15, "bold": True, "color": BLUE},
        {"text": "Each cell is the park's effect on pitchers, in wOBA points per ball in play, for one batter side "
                 "(L or R) and one kind of contact. Blue helps the pitcher; red hurts.", "size": 13, "space": 10},
        {"text": "Parks are ordered from most pitcher-friendly (top) to most hitter-friendly (bottom).", "size": 13, "space": 10},
        {"text": f"Average size of the effect: biggest for {biggest}; smallest for {smallest}.", "size": 13},
    ])


def _buy_low_slide(prs, summary):
    s = _new_slide(prs, "Buy-low free agents: results worse than the pitcher")
    t = summary.sort_values("undervalued_pts", ascending=False).head(10)
    rows = [["Pitcher", "Role", "Recent wOBA allowed", "Park-neutral", "Talent estimate", "Undervalued by (pts)",
             "Contact luck (pts)", "Best park (projected)"]]
    for _, r in t.iterrows():
        rows.append([_first_last(r["name"]), ROLE_NAMES.get(r["role"], r["role"]), _w(r["raw_woba"]), _w(r["neutral_woba"]),
                     _w(r["talent_woba"]), f"{r['undervalued_pts']:+.0f}", f"{r['contact_luck_pts']:+.0f}",
                     f"{r['best_park']} {_w(r['best_proj'])}"])
    _add_table(s, 0.4, 1.2, rows, [2.1, 0.7, 1.4, 1.2, 1.3, 1.5, 1.3, 3.4], font=11)
    _add_text(s, 0.5, 1.2 + 0.34 * len(rows) + 0.5, 12.3, 1.8, [
        {"text": "Undervalued by = recent wOBA allowed minus the talent estimate, in points. It removes the parks he pitched in "
                 "and pulls extreme results toward average.", "size": 12, "color": MUTED, "space": 4},
        {"text": "Contact luck = points per plate appearance that hitters gained on contact beyond what exit velocity and launch "
                 "angle warranted, after removing the park (positive = unlucky for the pitcher; includes defense).", "size": 12, "color": MUTED, "space": 4},
        {"text": "Contract cost is not in this data. Use it alongside a salary projection to find who is cheap relative to his talent.",
         "size": 12, "color": MUTED, "space": 4},
    ] + _calibration_lines(summary))


def _calibration_lines(summary):
    lines = []
    cal, par = _read("talent_calibration.csv"), _read("talent_params.csv")
    if cal is not None and par is not None and len(cal) == 4:
        r = dict(zip(cal["model"], cal["rmse_pts"]))
        base = r["Last season's raw wOBA allowed"]; ours = r["Calibrated (luck removed, shrunk)"]
        lines.append({"text": f"The talent estimate removes {par['luck_weight'].iloc[0]:.0%} of contact luck and blends in "
                              f"{int(par['shrink_pa'].iloc[0])} PA of league average, fitted on past seasons. Predicting next season's "
                              f"wOBA allowed, it misses by {ours:.0f} pts on held-out seasons vs {base:.0f} using last season as-is.",
                      "size": 12, "color": MUTED, "space": 4})
    if summary is not None and summary["undervalued_pts"].max() < 5:
        lines.append({"text": "No free agent in this class is clearly undervalued by his results; most of the table is "
                              "good pitchers regressing, not bargains.", "size": 12, "color": MUTED})
    return lines


def _park_slide(prs, park, ctx):
    effects, arch, overall, fa_proj, fa_low, seasons, fig_dir = (
        ctx["effects"], ctx["arch"], ctx["overall"], ctx["fa_proj"], ctx["fa_low"], ctx["seasons"], ctx["fig_dir"])
    n_seasons = seasons.get(park)
    low = n_seasons is not None and n_seasons < LOW_CONFIDENCE_SEASONS
    s = _new_slide(prs, park, subtitle=f"Low confidence: {n_seasons} seasons of data" if low else None)

    stem = park.replace(" ", "_")
    bb_path = Path(fig_dir) / f"{stem}_bb_slide.png"
    ar_path = Path(fig_dir) / f"{stem}_arch_slide.png"
    if plot_park_bb(effects, park, path=bb_path, figsize=(5.9, 3.2), compact=True) is not None:
        s.shapes.add_picture(str(bb_path), Inches(0.4), Inches(1.05), width=Inches(5.9))
    plt.close("all")

    # --- right panel text
    pts = overall[park]
    rank = int((overall > pts).sum()) + 1
    e = effects[(effects["park"] == park) & (effects["bip"] >= 200)]
    big = e.loc[e["pitcher_edge"].abs().idxmax()]
    big_pts = big["pitcher_edge"] * 1000
    items = [
        {"text": "What this park does", "size": 13, "bold": True, "color": BLUE, "space": 2},
        {"text": f"Overall: {pts:+.0f} wOBA points per ball in play for pitchers (rank {rank} of {len(overall)}; 1 = most pitcher-friendly).", "size": 11, "space": 2},
        {"text": f"Biggest effect: {BB_NAMES[big['bb_type']]} by {SIDE_NAMES[big['stand']]} batters, "
                 f"{abs(big_pts):.0f} points {'in the pitcher' + chr(39) + 's favor' if big_pts > 0 else 'against the pitcher'}.", "size": 11, "space": 8},
    ]
    if fa_proj is not None:
        t = fa_proj[fa_proj["park"] == park].sort_values("proj_woba").head(3)
        if len(t):
            items.append({"text": "Free agents with the lowest projected wOBA allowed here", "size": 13, "bold": True, "color": BLUE, "space": 2})
            for k, (_, r) in enumerate(t.iterrows(), 1):
                items.append({"text": f"{k}. {_first_last(r['name'])} ({ROLE_NAMES.get(r['role'], r['role'])}): {_w(r['proj_woba'])} projected, {_w(r['talent_woba'])} talent",
                              "size": 11, "space": 1})
            items[-1]["space"] = 8
    if fa_low is not None:
        t = fa_low[fa_low["park"] == park].sort_values("gain_pts", ascending=False).head(3)
        if len(t):
            items.append({"text": "Buy-low free agents here (recent results vs. projection)", "size": 13, "bold": True, "color": BLUE, "space": 2})
            for _, r in t.iterrows():
                items.append({"text": f"{_first_last(r['name'])} ({ROLE_NAMES.get(r['role'], r['role'])}): {_w(r['raw_woba'])} recent, "
                                      f"{_w(r['proj_woba'])} projected ({r['gain_pts']:+.0f} pts)", "size": 11, "space": 1})
    _add_text(s, 6.5, 1.05, 6.5, 3.4, items)

    # --- archetype strip
    if plot_park_archetypes_both(arch, park, path=ar_path, n=3, figsize=(12.5, 2.7)) is not None:
        s.shapes.add_picture(str(ar_path), Inches(0.4), Inches(4.35), width=Inches(12.5))
    plt.close("all")
    _add_text(s, 0.4, 7.08, 12.5, 0.35, [
        {"text": "Pitch bars: how much better this park suits the pitch than a typical pitch (wOBA points on contact). "
                 "Park effects are validated; the pitcher-specific fit is exploratory.", "size": 9, "color": MUTED}])


def _limitations_slide(prs, gain_known=False):
    s = _new_slide(prs, "Limitations and next steps")
    _add_text(s, 0.6, 1.3, 12.1, 5.7, [
        {"text": "What the model cannot do", "size": 18, "bold": True, "color": BLUE, "space": 6},
        {"text": "Weather, defense, umpires, and lineups are not modeled; some leak into the park effect.", "size": 15, "space": 4},
        {"text": "The pitcher-specific part of park fit is small (a few wOBA points)"
                 + (" and the backtest measured only a small gain from it." if gain_known else " and could not be confirmed in the backtest."),
         "size": 15, "space": 4},
        {"text": "Park effects use batter side and batted-ball type, not spray direction or the individual pitch.", "size": 15, "space": 4},
        {"text": "Oakland Coliseum and Steinbrenner Field are excluded (no team plays there now); Sutter Health Park has only 2 seasons.", "size": 15, "space": 4},
        {"text": "Projections use a judgment-call amount of shrinkage toward the league average and recent seasons only.", "size": 15, "space": 14},
        {"text": "Next steps", "size": 18, "bold": True, "color": BLUE, "space": 6},
        {"text": "Add contract projections to rank buy-low targets by value, not just talent.", "size": 15, "space": 4},
        {"text": "Add spray direction and per-pitch data, and test movers (pitchers who changed parks).", "size": 15, "space": 4},
        {"text": "Extend the same method to hitters and to trade targets.", "size": 15},
    ])


# ------------------------------------------------------------------ entry point
def build_deck(author: str = "", out_path=None):
    effects = _read("park_bb_effects.csv")
    arch = _read("archetype_park_fit.csv")
    if effects is None or arch is None:
        raise FileNotFoundError("Run notebooks 03 first: park_bb_effects.csv and archetype_park_fit.csv are missing.")
    fa_proj = _read("fa_targets_by_park.csv")
    fa_low = _read("fa_buy_low_by_park.csv")
    summary = _read("fa_buy_low_summary.csv")
    stability = _read("stability_results.csv")
    backtest = _read("backtest_results.csv")

    parks = sorted(arch["park"].unique())
    overall = _park_overall(effects, parks)
    seasons = {}
    if fa_proj is not None and "park_seasons" in fa_proj.columns:
        seasons = fa_proj.drop_duplicates("park").set_index("park")["park_seasons"].to_dict()
    low_conf = [p for p in parks if seasons.get(p, 99) < LOW_CONFIDENCE_SEASONS]
    fig_dir = Path(FIGURES) / "deck"

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    _title_slide(prs, author)
    _approach_slide(prs)
    _validation_slide(prs, stability, backtest)
    _overview_slide(prs, effects, overall, fig_dir, low_conf)
    _heatmap_slide(prs, effects, overall, fig_dir)
    if summary is not None:
        _buy_low_slide(prs, summary)
    ctx = {"effects": effects, "arch": arch, "overall": overall, "fa_proj": fa_proj,
           "fa_low": fa_low, "seasons": seasons, "fig_dir": fig_dir}
    for park in parks:
        _park_slide(prs, park, ctx)
    gain_known = backtest is not None and bool(backtest.iloc[0]["r2_gain_lo"] > 0)
    _limitations_slide(prs, gain_known)

    out = Path(out_path) if out_path else Path(ROOT) / "slides" / "ballpark_pitch_fit.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"Saved {len(prs.slides)} slides to {out}")
    return out
