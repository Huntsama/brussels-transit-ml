"""The four README figures, each in a light and a dark version (GitHub shows the one matching the viewer's theme).

    uv run python docs/readme_figures.py        -> docs/img/readme-<name>-{light,dark}.png

Reads the files the pipelines already write (no training): data/transit_delay/*.parquet,
outputs/grid_load/fair-comparison.csv, outputs/transit_delay/backtest.csv. One story per figure:
  strike    five weeks of city-wide excess wait: the 12 Oct strike stands out, the storm does not
  elia      day-ahead load error, all hours and public holidays: correcting Elia beats Elia
  backtest  STIB model vs the best simple rule, one row per test week
  heatmap   when Brussels waits longest: weekday x hour
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"

# Reference palette (dataviz skill), validated light + dark: model = blue (slot 1), reference = orange (slot 2)
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e6e5e0",
                  ours="#2a78d6", ref="#eb6834", base="#b9b8b2",
                  ramp=["#f3f7fd", "#cfe0f6", "#9fc2ee", "#6aa0e3", "#2a78d6", "#1d5aa8", "#123c72"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8f8e88", grid="#2e2e2c",
                 ours="#3987e5", ref="#d95926", base="#5c5b57",
                 ramp=["#1f2633", "#203a5c", "#24538a", "#2c6cb6", "#3987e5", "#78acf0", "#bcd6f8"]),
}


def style(t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["text2"], "xtick.color": t["text2"], "ytick.color": t["text2"],
        "axes.edgecolor": t["grid"], "axes.grid": False, "font.size": 10.5, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 26,
    })


def subtitle(ax, text, t):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=t["text2"], fontsize=10, va="bottom")


def save(fig, name, theme):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f"readme-{name}-{theme}.png", dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def strike(t, theme):
    c = pd.read_parquet(ROOT / "data" / "transit_delay" / "city_hourly.parquet").dropna(subset=["excess_wait_min"])
    c["hour"] = pd.to_datetime(c["hour"], utc=True).dt.tz_convert("Europe/Brussels")
    c = c[c["hour"].dt.hour.between(6, 22)]
    day = c.groupby(c["hour"].dt.date)["excess_wait_min"].mean()
    day.index = pd.to_datetime(day.index)
    day = day[day.index < day.index.max()]  # the last day is partial
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(day.index, day.values, color=t["ours"], lw=2, marker="o", ms=3.5, zorder=3)
    ax.axhline(day.median(), color=t["muted"], lw=1, ls=(0, (3, 3)))
    ax.text(day.index[0], day.median() + 0.12, f"typical day: {day.median():.1f} min", va="bottom", color=t["text2"], fontsize=9)
    s = day.idxmax()
    ax.annotate(f"National strike, {s:%a %d %b}\n{day.max():.1f} min late all day", (s, day.max()),
                xytext=(18, -4), textcoords="offset points", color=t["text"], fontsize=9.5, va="top",
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=1))
    storm = pd.Timestamp("2026-10-24")
    if storm in day.index:
        ax.annotate("3-day storm (23-25 Oct):\nno effect on STIB", (storm, day[storm]), xytext=(0, 42),
                    textcoords="offset points", ha="center", color=t["text2"], fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=1))
    ax.set_ylabel("Excess wait (min)")
    ax.yaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b"))
    ax.set_title("The lakehouse sees real events, without being told")
    subtitle(ax, f"Average extra wait at a Brussels STIB stop beyond the timetable, per day, 06:00-23:00 "
                 f"({day.index.min():%d %b} to {day.index.max():%d %b %Y})", t)
    save(fig, "strike", theme)


def elia(t, theme):
    f = pd.read_csv(ROOT / "outputs" / "grid_load" / "fair-comparison.csv").set_index("model")
    rows = [("Elia's own day-ahead forecast", "Elia day-ahead forecast", t["ref"]),
            ("LightGBM alone", "LightGBM, forecast weather issued D-2", t["base"]),
            ("LightGBM correcting Elia", "LightGBM + Elia as input, forecast weather issued D-2", t["ours"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, col, title in [(axes[0], "mae_mw", "All 22,870 test hours"), (axes[1], "holiday_mae_mw", "767 public-holiday hours")]:
        vals = [f.loc[k, col] for _, k, _ in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, color=[c for *_, c in rows], height=0.56)
        for yi, v in zip(y, vals):
            ax.text(v + 4, yi, f"{v:.0f} MW", va="center", color=t["text"], fontsize=9.5)
        ax.set_xlim(0, 360)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.text(0, 1.03, title, transform=ax.transAxes, color=t["text2"], fontsize=10)
        ax.tick_params(left=False)
    axes[0].set_yticks(np.arange(len(rows))[::-1], [r[0] for r in rows])
    gain = -f.loc[rows[2][1], "vs_elia_pct"]
    fig.suptitle(f"Correcting Elia's forecast cuts its error by {gain:.0f}%", x=0.02, ha="left", fontsize=13,
                 fontweight="bold", y=1.22, color=t["text"])
    fig.text(0.02, 1.04, "Mean absolute error of next-day Belgian electricity demand (lower is better), Mar 2024 to "
             "Oct 2026,\nusing only weather forecasts issued two days ahead, the information Elia had", ha="left",
             fontsize=9.5, color=t["text2"])
    save(fig, "elia", theme)


def backtest(t, theme):
    b = pd.read_csv(ROOT / "outputs" / "transit_delay" / "backtest.csv")
    b["week"] = pd.to_datetime(b["test_start"]).dt.strftime("%d %b") + " - " + pd.to_datetime(b["test_end"]).dt.strftime("%d %b")
    order = sorted(b["week"].unique(), key=lambda w: pd.to_datetime(w[:6] + " 2026"))
    series = [("Always 0 (trust the timetable)", "Trust the timetable", t["base"], "o"),
              ("Average per stop/line/hour of day", "Best simple rule", t["ref"], "s"),
              ("LightGBM without air/grid/city features", "LightGBM", t["ours"], "D")]
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.75 * len(order)))
    for i, w in enumerate(order):
        sub = b[b.week == w].set_index("model")["mae"]
        xs = [sub[m] for m, *_ in series]
        ax.plot([min(xs), max(xs)], [i, i], color=t["grid"], lw=2, zorder=1)
        for (m, label, col, mk), x in zip(series, xs):
            ax.scatter(x, i, s=70, color=col, marker=mk, zorder=3, edgecolor=t["surface"], linewidth=1.5)
        gain = 100 * (1 - sub[series[2][0]] / sub[series[1][0]])
        ax.text(min(xs) - 0.012, i, f"{gain:.1f}% better", ha="right", va="center", color=t["text"], fontsize=9.5)
    ax.set_yticks(range(len(order)), [f"Test week {w}" for w in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    lo = b[b.model.isin([s[0] for s in series])]["mae"].min()
    ax.set_xlim(lo - 0.12, b["mae"].max() * 0 + max(b[b.model == series[0][0]]["mae"]) + 0.04)
    ax.set_xlabel("Mean absolute error, minutes of extra wait (lower is better)")
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker=mk, color=col, ms=8, label=label) for _, label, col, mk in series]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              labelcolor=t["text"], fontsize=9.5)
    ax.set_title("Next-hour STIB delays: the model wins every test week", pad=40)
    save(fig, "backtest", theme)


def heatmap(t, theme):
    f = pd.read_parquet(ROOT / "data" / "transit_delay" / "features.parquet",
                        columns=["dow", "hod", "excess_wait_min", "disrupted"])
    f = f[~f["disrupted"] & f["hod"].between(6, 23)]  # 05h = first departures, few vehicles: it would swamp the scale
    g = f.groupby(["dow", "hod"])["excess_wait_min"].mean().unstack()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", t["ramp"])
    fig, ax = plt.subplots(figsize=(9, 3.1))
    im = ax.imshow(g.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(g.index)])
    ax.set_xticks(range(0, len(g.columns), 2), [f"{h:02d}h" for h in g.columns[::2]])
    ax.tick_params(left=False, bottom=False)
    for s in ax.spines.values():
        s.set_visible(False)
    i, j = np.unravel_index(np.nanargmax(g.values), g.shape)
    ax.text(j, i, f"{g.values[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
            color="#ffffff" if theme == "light" else "#0b0b0b")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["text2"], length=0)
    cb.set_label("min", color=t["text2"])
    ax.set_title("When Brussels waits longest")
    subtitle(ax, "Average extra wait beyond the timetable per stop and line, by weekday and hour "
                 "(five weeks, 06:00-23:00, strike day left out)", t)
    save(fig, "heatmap", theme)


if __name__ == "__main__":
    for theme, t in THEMES.items():
        style(t)
        for fn in (strike, elia, backtest, heatmap):
            fn(t, theme)
    print("\n".join(sorted(p.name for p in IMG.glob("readme-*.png"))))
