# Title: Global Hydrogen Trade and Investment Modelling: Business Strategies for Adoption
# Author: William Cope
# University: University of Cambridge - Department of Chemical Engineering and Biotechnology
# Degree: MPhil in Advanced Chemical Engineering
# Date: 2025-08-22
#
# Notes:

# ============================== Imports =============================== #

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from math import isfinite

# Optional geo imports (graceful fallback)
try:
    import geopandas as gpd
    GEO_OK = True
except Exception:
    GEO_OK = False

def _load_world_gdf():
# Return a GeoDataFrame of world countries if possible, else None.
    if not GEO_OK:
        return None
    try:
        import geodatasets as gd  # pip install geodatasets
        path = gd.get_path("naturalearth.countries")
        return gpd.read_file(path)
    except Exception:
        return None

# ============================= Plot Style ============================= #
OKABE_ITO = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#999999"]

# Accessible palettes (Paul Tol) + neutrals
TOL_BRIGHT = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
TOL_MUTED  = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

GREEN_STRONG = "#2ca02c"
RED_STRONG   = "#d62728"
GRAY_DARK    = "#444444"
GRAY_MED     = "#666666"
GRAY_LIGHT   = "#bbbbbb"

# === Custom palettes from screenshots ===
YEAR_COLORS = ["#3594cc", "#ea801c", "#8cc5e3", "#f0b077"]  # Med Blue, Med Orange, Light Blue, Light Orange

TORNADO_COLORS = ["#0d7d87", "#c31e23", "#99c6cc"]          # Dark Teal, Dark Red, Light Teal

BLUE_YELLOW = {"blue": "#1a80bb", "yellow": "#f2c45f"}      # HHI vs Band

def use_cambridge_style(usetex: bool = False):
    if usetex:
        mpl.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{lmodern}\usepackage[T1]{fontenc}"
        })
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "TeX Gyre Termes", "Times New Roman"],
        "mathtext.fontset": "cm",
        "mathtext.default": "regular",
        "axes.unicode_minus": True,

        # larger canvas so labels are readable but duals don’t clash
        "figure.figsize": (10.8, 6.8),
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "lines.linewidth": 2.2,
        "lines.markersize": 6.2,

        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,

        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.edgecolor": "black",

        # legends: always square corners, consistent
        "legend.frameon": True,
        "legend.fancybox": False,   # <— square corners everywhere
        "legend.fontsize": 15,

        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def autosave_show(fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    counter = {"i": 1 + len(list(fig_dir.glob("fig_*.pdf")))}
    def _show(*_, **__):
        i = counter["i"]
        stem = f"fig_{i:03d}"
        fig = plt.gcf()
        fig.savefig(fig_dir / f"{stem}.pdf")
        fig.savefig(fig_dir / f"{stem}.png", dpi=300)
        counter["i"] += 1
        plt.close(fig)
    plt.show = _show

# ============================== Loaders ============================== #
def load_run(run_path: str | Path | None = None):
    exports = Path("exports")
    if run_path is None:
        runs = sorted([p for p in exports.glob("run_*") if p.is_dir()])
        if not runs:
            raise SystemExit("No runs found in ./exports. Run your model once to create an export.")
        run_path = runs[-1]
    run_path = Path(run_path)
    manifest_fp = run_path / "manifest.json"
    if not manifest_fp.exists():
        raise SystemExit(f"No manifest.json found in {run_path}")
    with manifest_fp.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    def load_entry(name: str):
        meta = manifest["files"][name]
        fp = run_path / meta["path"]
        if meta["type"] == "dataframe":
            try:
                return pd.read_parquet(fp)
            except Exception as e:
                raise SystemExit(
                    f"Failed to read {fp} — install parquet support:\n"
                    f"  py -m pip install pyarrow\nOriginal error: {e}"
                )
        with fp.open("r", encoding="utf-8") as jf:
            return json.load(jf)

    data = {name: load_entry(name) for name in manifest["files"]}
    data["_dir"] = str(run_path)
    return data, manifest

def print_loaded_summary(data: dict, max_cols: int = 14):
    print("\n=== Loaded datasets summary ===")
    for k, v in data.items():
        if k == "_dir": continue
        if isinstance(v, pd.DataFrame):
            cols = list(map(str, v.columns))
            extra = f" (+{len(cols)-max_cols} more)" if len(cols) > max_cols else ""
            print(f" • {k}: DataFrame with {len(v):,} rows, cols: {cols[:max_cols]}{extra}")
        elif isinstance(v, dict):
            print(f" • {k}: dict keys(sample)={list(v.keys())[:8]}")
        else:
            print(f" • {k}: {type(v).__name__}")
    print("=== End summary ===\n")

# ============================== Helpers ============================== #
# Small utilities shared by multiple figures (formatters, layout helpers)

# Save a single figure
def _save_named(fig, fig_dir: Path, stem: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf")
    fig.savefig(fig_dir / f"{stem}.png", dpi=300)
    plt.close(fig)

def box_axes(ax, lw: float = 1.1):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)

def legend_outside_top(ax, ncol: int | None = None, y: float = 1.02, top_pad: float = 0.94):
    handles, labels = ax.get_legend_handles_labels()
    if not handles: return
    if ncol is None: ncol = min(4, max(1, (len(labels)+1)//2))
    leg = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
        frameon=True, fancybox=False,  # <— force square corners
        edgecolor="black", borderpad=0.5, labelspacing=0.6, handlelength=2.2
    )
    leg.get_frame().set_linewidth(1.1)
    ax.figure.subplots_adjust(top=top_pad)

def add_group_titles_top(ax, groups, fontsize=12):
    for (start_idx, end_idx, title) in groups:
        # Place centred between provided integer slot indices (inclusive)
        xmid = (start_idx + end_idx) / 2.0
        ax.text(xmid, 1.02, title, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=fontsize)

def _shorten(name, n=16):
    name = str(name)
    return name if len(name) <= n else name[:n-3] + "..."

def find_df_with_cols(data: dict, required_cols: set[str]) -> tuple[str, pd.DataFrame] | None:
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            cols = set(map(str, v.columns))
            if required_cols.issubset(cols):
                return k, v
    return None

def _percent_formatter01(ax):
# Percent formatter for [0,1] axes.
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))

# Gini coefficient on finite values only robust to zeros and NaNs.
def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]  # drop non-finite values
    if x.size == 0: return np.nan  # undefined on empty input
    if np.min(x) < 0: x = x - np.min(x)  # shift if negative values present
    if x.sum() == 0: return 0.0  # all zeros => no inequality
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Standard discrete Gini formula (Lorenz-based)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# Grouped bar helper with consistent edge styling and centred offsets.
def grouped_bars(ax, series, categories, series_labels=None,
                 colors=None, ylabel="Value", bar_w=0.16, gap=0.02):
    series = [np.asarray(s) for s in series]
    n_series = len(series); x = np.arange(len(categories), dtype=float)
    colors = colors or OKABE_ITO[:n_series]
    series_labels = series_labels or [f"Series {i+1}" for i in range(n_series)]
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * (bar_w + gap)  # centre groups around tick
    for i in range(n_series):
        y = series[i]
        ax.bar(x + offsets[i], y, width=bar_w, color=colors[i], edgecolor="black", linewidth=1.0, label=series_labels[i])
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=0)
    ax.set_ylabel(ylabel); box_axes(ax)
    return offsets  # return offsets so callers can draw overlays aligned to bars

# === Market code -> City
CITY_FROM_MARKET = {
    "JPN_KAW": "Kawasaki",
    "KOR_ULS": "Ulsan",
    "GER_WIL": "Wilhelmshaven",
    "NLD_ROT": "Rotterdam",
    "BEL_ANR": "Antwerp",
    "CHN_SHG": "Shanghai",
    "IND_GUJ": "Gujarat",
    "UK_TEE": "Teesside",
    "UK_TEE_SUP": "Teesside",
    "AUS_GLD": "Gladstone",
    "AUS_PIL": "Pilbara",
    "CHL_ANT": "Antofagasta",
    "SAU_NEOM": "Neom",
    "OMN_DUQ": "Duqm",
    "UAE_RUW": "Ruwais",
    "NAM_LUD": "Luderitz",
    "RSA_BOE": "Boegoebaai",
    "MAR_SOU": "Souss-Massa",
    "MRT_NDB": "Nouadhibou",
    "EGY_SUE": "Suez",
    "KAZ_MNG": "Mangystau",
    "USA_HOU": "Houston",
    "CAN_PTT": "Point Tupper",
    "NOR_BER": "Bergen",
}

SUFFIX_TO_CITY = {
    "KAW": "Kawasaki", "ULS": "Ulsan", "WIL": "Wilhelmshaven", "ROT": "Rotterdam",
    "ANR": "Antwerp", "SHG": "Shanghai", "TEE": "Teesside", "GLD": "Gladstone",
    "PIL": "Pilbara", "ANT": "Antofagasta", "NEOM": "Neom", "DUQ": "Duqm",
    "RUW": "Ruwais", "LUD": "Luderitz", "BOE": "Boegoebaai", "SOU": "Souss-Massa",
    "NDB": "Nouadhibou", "SUE": "Suez", "MNG": "Mangystau", "HOU": "Houston",
    "PTT": "Point Tupper", "BER": "Bergen",
}

def market_to_city(m) -> str:
    if isinstance(m, str):
        if m in CITY_FROM_MARKET:
            return CITY_FROM_MARKET[m]
        if "_" in m:
            suf = m.split("_")[-1]
            return SUFFIX_TO_CITY.get(suf, m)
    return str(m)

# ============================= Wrangling ============================= #
def _delivered_df(data):
    delivered = data.get("delivered_prices_det")
    if delivered is None: return None
    delivered = {str(k): v for k, v in delivered.items()}
    years = sorted(int(y) for y in delivered.keys())  # ensure chronological order
    markets = sorted({m for y in years for m in delivered[str(y)].keys()})
    rows = [{"Year": y, "Market": m, "Price": delivered[str(y)].get(m, np.nan)}
            for y in years for m in markets]
    return pd.DataFrame(rows)

def _price_stats(df):
    out = []
    for y, sub in df.groupby("Year"):
        s = sub["Price"].dropna()
        if s.empty: continue
        q = s.quantile([0.1,0.25,0.5,0.75,0.9])  # robust percentiles for spread
        p10,p25,p50,p75,p90 = q.iloc[0],q.iloc[1],q.iloc[2],q.iloc[3],q.iloc[4]
        out.append(dict(Year=y, P10=p10, P25=p25, Median=p50, P75=p75, P90=p90,
                        IQR=(p75-p25), Band=(p90-p10), Mean=s.mean(), Gini=gini(s.values)))
    return pd.DataFrame(out).sort_values("Year")

# ============================= Axis Ticks ============================ #
def _round_step(mag):
# Choose a nice rounding step based on magnitude.
    if mag >= 200: return 50
    if mag >= 100: return 20
    if mag >= 50:  return 10
    if mag >= 10:  return 2
    if mag >= 2:   return 0.5
    return 0.1


def _nice_values(vmin: float, vmax: float, n: int = 5, fixed_step: float | None = None):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return [0]*n
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin if vmax > vmin else (abs(vmax) if vmax else 1.0)
    step = span / (n-1)
    step = fixed_step if fixed_step is not None else _round_step(max(abs(vmin), abs(vmax), step))
    first = np.floor(vmin / step) * step
    last = np.ceil(vmax / step) * step
    vals = np.linspace(first, last, n)
    return [float(np.round(v/step)*step) for v in vals]

# ============================= Figures =============================== #

# F1 — Distributions by year
# Violin plots of price distributions by year
def F1_price_violins(data):
    df = _delivered_df(data)
    if df is None or df.empty: return
    stats = _price_stats(df); years = stats["Year"].tolist()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    # Violin shapes per year; overlay median and IQR for interpretability.
    parts = ax.violinplot([df[df["Year"] == y]["Price"].dropna().values for y in years],
                          positions=np.arange(len(years))+1, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("#d9d9d9"); pc.set_edgecolor("none"); pc.set_alpha(0.95)
    ax.plot(np.arange(len(years))+1, stats["Median"], marker="o", lw=1.8, label="Median", color=TOL_BRIGHT[4])
    ax.fill_between(np.arange(len(years))+1, stats["P25"], stats["P75"], alpha=0.25, label="IQR (P25-P75)", color=TOL_BRIGHT[4])
    ax.plot(np.arange(len(years))+1, stats["P10"], ls="--", label="P10", color=GREEN_STRONG)
    ax.plot(np.arange(len(years))+1, stats["P90"], ls="--", label="P90", color=RED_STRONG)

    ax.set_xticks(np.arange(len(years))+1); ax.set_xticklabels(years)
    ax.set_xlabel("Year"); ax.set_ylabel("Price [$/kg]")
    box_axes(ax); legend_outside_top(ax, ncol=4)
    plt.show()

# F3 — ECDF
# Empirical cumulative distribution functions by year
def F3_ecdf(data):
    df = _delivered_df(data)
    if df is None or df.empty: return
    years = sorted(df["Year"].unique())
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for i, y in enumerate(years):
        s = df[df["Year"] == y]["Price"].dropna().sort_values()
        if s.empty: continue
        x = s.values; F = np.arange(1, len(x)+1) / len(x)  # empirical CDF
        ax.step(x, F, where="post", label=str(y),
                color=YEAR_COLORS[i % len(YEAR_COLORS)])

    ax.set_xlabel("Price [$/kg]"); ax.set_ylabel("Cumulative share [%]")
    _percent_formatter01(ax)
    box_axes(ax); legend_outside_top(ax, ncol=4); plt.show()

# Multi-year exemplar nodes (low/mid/high deciles)
def F4_exemplars_all_years_clean(data):
    df = _delivered_df(data)
    if df is None or df.empty: return
    df["Market"] = df["Market"].astype(str)
    years = sorted(df["Year"].unique())
    if len(years) < 2: return
    y0, y1 = years[0], years[-1]

    base = df[df["Year"] == y0].dropna(subset=["Price"]).copy()
    if base.empty: return

    # Only retain markets with finite values in every displayed year
    def present_all_years(m):
        for y in years:
            v = df[(df["Year"]==y)&(df["Market"]==m)]["Price"].values
            if v.size==0 or not np.isfinite(float(v[0])): return False
        return True

    base_sorted_low  = base.sort_values("Price", ascending=True)["Market"].tolist()
    base_sorted_high = base.sort_values("Price", ascending=False)["Market"].tolist()
    medv = base["Price"].median()
    base["absdiff"] = (base["Price"] - medv).abs()
    base_sorted_mid  = base.sort_values("absdiff")["Market"].tolist()

    q_low  = base["Price"].quantile(0.1)
    q_high = base["Price"].quantile(0.9)
    bottom_pool = [m for m in base_sorted_low if base.set_index("Market").loc[m, "Price"] <= q_low]
    top_pool    = [m for m in base_sorted_high if base.set_index("Market").loc[m, "Price"] >= q_high]
    mid_pool    = [m for m in base_sorted_mid if (base.set_index("Market").loc[m, "Price"]>q_low and base.set_index("Market").loc[m, "Price"]<q_high)]

    def pick_group(pool_primary, order_widen, need, taken):
        picks=[]
        for m in pool_primary:
            if m in taken or m in picks: continue
            if present_all_years(m): picks.append(m)
            if len(picks)==need: return picks
        for m in order_widen:
            if m in taken or m in picks: continue
            if present_all_years(m): picks.append(m)
            if len(picks)==need: return picks
        remaining = [m for m in base["Market"].tolist() if m not in taken and m not in picks]
        for m in remaining:
            if present_all_years(m): picks.append(m)
            if len(picks)==need: return picks
        for m in remaining:
            picks.append(m)
            if len(picks)==need: return picks
        return picks

    picks_top = pick_group(top_pool, base_sorted_high, 3, set())  # high-priced exemplars
    picks_mid = pick_group(mid_pool, base_sorted_mid, 3, set(picks_top))  # near-median exemplars
    picks_bot = pick_group(bottom_pool, base_sorted_low, 3, set(picks_top + picks_mid))  # low-priced exemplars
    picks = (picks_top + picks_mid + picks_bot)[:9]

    # Show full city names
    cats = [market_to_city(m) for m in picks]

    series = []
    ymax = 0.0
    for y in years:
        yy=[]
        for m in picks:
            v = df[(df["Year"]==y)&(df["Market"]==m)]["Price"]
            val = float(v.values[0]) if not v.empty and isfinite(v.values[0]) else np.nan
            yy.append(val)
            if isfinite(val): ymax = max(ymax, val)
        series.append(yy)

    fig, ax = plt.subplots(figsize=(14.2, 6.9))
    year_labels = [str(y) for y in years]
    bar_colors  = [YEAR_COLORS[i % len(YEAR_COLORS)] for i in range(len(years))]

    bar_w = 0.14; gap = 0.02
    offsets = grouped_bars(ax, series, cats, series_labels=year_labels,
                           colors=bar_colors, ylabel="Price [$/kg]", bar_w=bar_w, gap=gap)

        # Use one consistent light grey for separators & gradient/leader
    DIV_GREY = "#e6e6e6"

    # Faint separators between cities
    for i in range(len(picks)-1):
        ax.axvline(i + 0.5, color=DIV_GREY, lw=1.0, zorder=0)

    ax.set_ylim(top=ymax * 1.24 if ymax>0 else None)
    ax.set_xlabel("Location")

    # Dotted gradient line first->last year per city + raised % labels
    idx0, idx1 = 0, len(years)-1
    ymin_ax, ymax_ax = ax.get_ylim()
    raise_frac = 0.24  # higher than before for clear separation

    for i, m in enumerate(picks):
        v0 = df[(df["Year"]==y0)&(df["Market"]==m)]["Price"]
        v1 = df[(df["Year"]==y1)&(df["Market"]==m)]["Price"]
        if not v0.empty and not v1.empty and isfinite(v0.values[0]) and isfinite(v1.values[0]) and v0.values[0] != 0:
            x0 = i + offsets[idx0]; y0v = float(v0.values[0])
            x1 = i + offsets[idx1]; y1v = float(v1.values[0])

            # dotted gradient line in the same grey as the separators
            ax.plot([x0, x1], [y0v, y1v], lw=1.4, ls=":", color="#000000", zorder=3)

            xmid = 0.5*(x0+x1); ymid = 0.5*(y0v+y1v)
            pct = 100.0*(y1v - y0v)/y0v
            ytext = min(ymid + raise_frac*(ymax_ax - ymin_ax), ymax_ax - 0.10*(ymax_ax - ymin_ax))

            # vertical dotted leader in the same grey
            ax.annotate(f"{pct:+.1f}%", xy=(xmid, ymid), xytext=(xmid, ytext),
                        ha="center", va="bottom", fontsize=14, color=GRAY_DARK,
                        arrowprops=dict(arrowstyle="-", linestyle=":", color=DIV_GREY, lw=1.4))

    legend_outside_top(ax, ncol=min(5, len(series)), y=1.02, top_pad=0.94)
    plt.show()
    return picks

# F7 — Concentration vs price dispersion
# Compare export concentration (HHI) with price dispersion
def F7_concentration_vs_dispersion(data):
    df_trade = data.get("df_trade_det")
    df_deliv = _delivered_df(data)
    if df_trade is None or df_deliv is None or not {"Year","From"}.issubset(df_trade.columns): return
    flow_col = "Flow_kt" if "Flow_kt" in df_trade.columns else ([c for c in df_trade.columns if "Flow" in c][0] if any("Flow" in c for c in df_trade.columns) else None)
    if flow_col is None: return
    hh = []
    for y, sub in df_trade.groupby("Year"):
        s = sub.groupby("From")[flow_col].sum()
        if s.sum() == 0: continue
        share = (s/s.sum())**2  # Herfindahl components
        hh.append((y, float(share.sum()*10000)))
    if not hh: return
    years_h, hhis = zip(*sorted(hh))
    stats = _price_stats(df_deliv)
    disp = stats.set_index("Year")["Band"].reindex(years_h)
    fig, ax1 = plt.subplots(figsize=(9.8, 5.4))
    ax1.plot(years_h, hhis, "-o", color=BLUE_YELLOW["blue"], label="Export HHI")
    ax1.set_ylabel("HHI (index)")
    box_axes(ax1)
    ax2 = ax1.twinx()
    ax2.plot(years_h, disp.values, "--s", color=BLUE_YELLOW["yellow"], label="P90-P10 price band")
    ax2.set_ylabel("Price dispersion [$/kg]")
    box_axes(ax2)

    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(h1+h2, l1+l2, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0); ax1.figure.subplots_adjust(top=0.92)
    plt.show()

# F26 — Portfolio scatter (cividis colormap)
# Scatter of NPV vs probability of finance (colour=median IRR)
def F26_portfolio_scatter(data):
    df_mc = data.get("df_mc")
    wacc = data.get("wacc", None)
    if df_mc is None or not {"Sector","IRR%","NPV $M"}.issubset(df_mc.columns) or not isinstance(wacc, (int, float)):
        return
    agg = df_mc.groupby("Sector", observed=True).agg(
        ExpNPV=("NPV $M", "mean"),
        SDNPV=("NPV $M", "std"),
        MedIRR=("IRR%", "median"),
        ProbFinance=("IRR%", lambda s: (s >= wacc).mean()*100)
    ).reset_index()
    sizes = np.clip(np.abs(agg["ExpNPV"].to_numpy()), 1, None)
    sizes = 50 + 250 * (sizes / np.nanmax(sizes))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    sc = ax.scatter(
        agg["ProbFinance"], agg["ExpNPV"],
        s=sizes, c=agg["MedIRR"], cmap="cividis",
        edgecolor="black"
    )
    for _, r in agg.iterrows():
        ax.text(r["ProbFinance"]+0.5, r["ExpNPV"], _britify(_shorten(r["Sector"], 14)), fontsize=10, va="center", color=GRAY_DARK)

    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Median IRR [%]")
    ax.set_xlabel("Pr(IRR >= WACC) [%]"); ax.set_ylabel("Expected NPV [USD million]")

    box_axes(ax); plt.show()

# helpers for sensitivities
def _std_beta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 3: return np.nan
    x_std = (x - x.mean())/x.std() if x.std() else x - x.mean()
    y_std = (y - y.mean())/y.std() if y.std() else y - y.mean()
    b = np.polyfit(x_std, y_std, 1)[0]
    return float(b)

def DASH_Sector_sensitivity_all(data, bins_price=8, bins_carbon=8):
    df = data.get("df_mc")
    if df is None or "Sector" not in df.columns: return
    sectors = sorted(df["Sector"].dropna().unique().tolist())
    for sec in sectors:
        fig, axs = plt.subplots(1, 2, figsize=(14.2, 6.0))
        sub = df[df["Sector"] == sec].copy()

        # Tornado
                # Tornado
        need = {"IRR%","H2 Price","Carbon $/t","Subsidy $/kg"}
        if need.issubset(sub.columns):
            vals = []
            for (var, col) in [("H2 Price", TORNADO_COLORS[0]),
                               ("Carbon $/t", TORNADO_COLORS[1]),
                               ("Subsidy $/kg", TORNADO_COLORS[2])]:
                ...

                b = _std_beta(sub[var], sub["IRR%"])
                vals.append((_shorten(var, 18), 0 if (b is None or not np.isfinite(b)) else abs(b), col))
            vals = sorted(vals, key=lambda t: t[1], reverse=True)
            labs = [v[0] for v in vals]; bet = [v[1] for v in vals]; cols = [v[2] for v in vals]
            axs[0].barh(np.arange(len(bet)), bet, color=cols, edgecolor="black")
            axs[0].set_yticks(np.arange(len(bet))); axs[0].set_yticklabels(labs)
            axs[0].invert_yaxis(); axs[0].set_xlabel("|beta|")
            box_axes(axs[0])
        else:
            axs[0].axis("off")

        # Heatmap with quantile bins (minimises empty/grey cells)
        need2 = {"IRR%","H2 Price","Carbon $/t"}
        if need2.issubset(sub.columns):
            sub2 = sub[np.isfinite(sub["H2 Price"]) & np.isfinite(sub["Carbon $/t"]) & np.isfinite(sub["IRR%"])].copy()
            if not sub2.empty:
                # quantile edges (drop dup edges if data is discrete)
                qx = np.linspace(0, 1, bins_price+1)
                qy = np.linspace(0, 1, bins_carbon+1)
                p_edges = pd.Series(sub2["H2 Price"]).quantile(qx, interpolation="linear").values  # quantile bins balance counts
                c_edges = pd.Series(sub2["Carbon $/t"]).quantile(qy, interpolation="linear").values  # mitigate sparsity
                p_edges = np.unique(p_edges); c_edges = np.unique(c_edges)
                if len(p_edges) < 3 or len(c_edges) < 3:
                    # fallback to linear if data too flat
                    p_edges = np.linspace(sub2["H2 Price"].min(), sub2["H2 Price"].max(), bins_price+1)
                    c_edges = np.linspace(sub2["Carbon $/t"].min(), sub2["Carbon $/t"].max(), bins_carbon+1)

                sub2["P_bin"] = pd.cut(sub2["H2 Price"], p_edges, include_lowest=True, duplicates="drop")
                sub2["C_bin"] = pd.cut(sub2["Carbon $/t"], c_edges, include_lowest=True, duplicates="drop")

                pivot = sub2.groupby(["P_bin","C_bin"], observed=True)["IRR%"].median().unstack()
                arr = pivot.values.astype(float)
                arr = np.where(np.isfinite(arr), arr, np.nan)

                cmap = mpl.cm.viridis.copy()
                cmap.set_bad("#efefef")  # very light for any rare no-data cell
                im = axs[1].imshow(np.ma.masked_invalid(arr), aspect="auto", origin="lower", cmap=cmap)

                # Clean numeric axes (min–…–max), independent of bins
                pmin, pmax = float(sub2["H2 Price"].min()), float(sub2["H2 Price"].max())
                cmin, cmax = float(sub2["Carbon $/t"].min()), float(sub2["Carbon $/t"].max())
                ncols, nrows = pivot.shape[1], pivot.shape[0]
                axs[1].set_xticks(np.linspace(0, ncols-1, 5))
                axs[1].set_yticks(np.linspace(0, nrows-1, 5))
                xtlabs = _nice_values(pmin, pmax, n=5)
                ytlabs = _nice_values(cmin, cmax, n=5)
                axs[1].set_xticklabels([f"{v:.1f}" if abs(v)<100 else f"{v:.0f}" for v in xtlabs])
                axs[1].set_yticklabels([f"{v:.1f}" if abs(v)<100 else f"{v:.0f}" for v in ytlabs])

                axs[1].set_xlabel("H2 price [$/kg]")
                axs[1].set_ylabel("Carbon price [$/t]")
                cbar = plt.colorbar(im, ax=axs[1]); cbar.set_label("IRR [%]")
                box_axes(axs[1])
            else:
                axs[1].axis("off")
        else:
            axs[1].axis("off")
        plt.tight_layout(); plt.show()

# Grid of heatmaps for sector IRR surfaces
def Combined_Heatmaps_All_Sectors(data, bins_price=8, bins_carbon=8, share_scale=True):
# 
#     Five heatmaps in a tight 2–2–1 layout (last centred), all SQUARE, big fonts.//
#     Uses a skinny rightmost column for the shared colorbar so plots sit close!
#     
    df = data.get("df_mc")
    if df is None or "Sector" not in df.columns:
        return

    sectors = sorted(df["Sector"].dropna().unique().tolist())[:5]
    if not sectors:
        return

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(36, 24))  # big & readable
    # 5 columns: 4 for plots, 1 skinny for the colorbar
    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.06],  # skinny colorbar at col=4
        height_ratios=[1, 1, 1],
        wspace=0.02,   # same-row plots very close
        hspace=0.32    # extra space between rows to avoid title/xlabel collisions
    )

    # slots for 2–2–1 (last centred under the top four)
    slots = [
        (0, slice(0, 2)),  # row 0, cols 0–1
        (0, slice(2, 4)),  # row 0, cols 2–3
        (1, slice(0, 2)),  # row 1, cols 0–1
        (1, slice(2, 4)),  # row 1, cols 2–3
        (2, slice(1, 3)),  # row 2, centred: cols 1–2
    ]
    cax = fig.add_subplot(gs[:, 4])  # shared colorbar axis (all rows)

    # --- Big, readable fonts
    FS_TITLE = 36
    FS_LABEL = 34
    FS_TICK  = 30
    FS_CBAR  = 34
    TITLE_PAD, XLABEL_PAD, YLABEL_PAD = 22, 20, 14


    vmin = vmax = None
    if share_scale and {"IRR%","H2 Price","Carbon $/t"}.issubset(df.columns):
        s = df["IRR%"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size:
            vmin, vmax = np.nanpercentile(s, [5, 95])

    # ---------- helpers kept INSIDE so they can see FS_* ----------
    def _nice_ticks(vmin_, vmax_, step, max_n=6):
        import math
        if vmax_ <= vmin_ or step <= 0:
            return np.array([vmin_, vmax_], dtype=float)
        start = math.floor(vmin_ / step) * step
        end   = math.ceil (vmax_ / step) * step
        ticks = np.arange(start, end + 0.5*step, step, dtype=float)
        if len(ticks) > max_n:
            idx = np.linspace(0, len(ticks)-1, max_n).round().astype(int)
            ticks = ticks[idx]
            ticks[0] = start; ticks[-1] = end
        return ticks

    def _format_ticks(ax, pmin, pmax, cmin, cmax, ncols, nrows,
                      p_step=0.5, c_step=25, max_n=6):
# Clean round ticks mapped to the imshow grid.
        xticks_val = _nice_ticks(pmin, pmax, p_step, max_n=max_n)
        yticks_val = _nice_ticks(cmin, cmax, c_step, max_n=max_n)

        def to_pos(vals, v0, v1, n):
            if v1 == v0:
                return np.zeros_like(vals)
            return ((vals - v0) / (v1 - v0)) * (n - 1)

        xlocs = to_pos(xticks_val, pmin, pmax, ncols)
        ylocs = to_pos(yticks_val, cmin, cmax, nrows)

        ax.set_xticks(xlocs); ax.set_yticks(ylocs)
        ax.set_xticklabels([f"{v:.1f}" for v in xticks_val], fontsize=FS_TICK)
        ax.set_yticklabels([f"{int(round(v))}" for v in yticks_val], fontsize=FS_TICK)

    def heat_for_sector(ax, sec):
        sub = df[df["Sector"] == sec]
        need = {"IRR%","H2 Price","Carbon $/t"}
        if not need.issubset(sub.columns):
            ax.axis("off"); return None

        sub2 = sub[np.isfinite(sub["H2 Price"]) &
                   np.isfinite(sub["Carbon $/t"]) &
                   np.isfinite(sub["IRR%"])].copy()
        if sub2.empty:
            ax.axis("off"); return None

        # Quantile bins (fallback to linear if too few unique edges)
        qx = np.linspace(0, 1, bins_price + 1)
        qy = np.linspace(0, 1, bins_carbon + 1)
        p_edges = pd.Series(sub2["H2 Price"]).quantile(qx, interpolation="linear").values  # quantile bins balance counts
        c_edges = pd.Series(sub2["Carbon $/t"]).quantile(qy, interpolation="linear").values  # mitigate sparsity
        p_edges = np.unique(p_edges); c_edges = np.unique(c_edges)
        if len(p_edges) < 3 or len(c_edges) < 3:
            p_edges = np.linspace(sub2["H2 Price"].min(), sub2["H2 Price"].max(), bins_price + 1)
            c_edges = np.linspace(sub2["Carbon $/t"].min(), sub2["Carbon $/t"].max(), bins_carbon + 1)

        sub2["P_bin"] = pd.cut(sub2["H2 Price"], p_edges, include_lowest=True, duplicates="drop")
        sub2["C_bin"] = pd.cut(sub2["Carbon $/t"], c_edges, include_lowest=True, duplicates="drop")

        pivot = sub2.groupby(["P_bin","C_bin"], observed=True)["IRR%"].median().unstack()
        arr = pivot.values.astype(float)
        arr = np.where(np.isfinite(arr), arr, np.nan)

        cmap = mpl.cm.viridis.copy(); cmap.set_bad("#efefef")
        im = ax.imshow(np.ma.masked_invalid(arr), origin="lower", aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax)

        # Keep square and fill slot
        try: ax.set_box_aspect(1)
        except Exception: pass
        ax.set_adjustable("box")

        # Clean, round numeric ticks (H2: 0.5 step; Carbon: 25 step)
        pmin, pmax = float(sub2["H2 Price"].min()), float(sub2["H2 Price"].max())
        cmin, cmax = float(sub2["Carbon $/t"].min()), float(sub2["Carbon $/t"].max())
        nrows, ncols = arr.shape
        _format_ticks(ax, pmin, pmax, cmin, cmax, ncols, nrows,
                      p_step=0.5, c_step=25, max_n=6)

        ax.set_xlabel("H2 price [$/kg]", fontsize=FS_LABEL, labelpad=XLABEL_PAD)
        ax.set_ylabel("Carbon price [$/t]", fontsize=FS_LABEL, labelpad=YLABEL_PAD)
        ax.set_title(_britify(_shorten(sec, 36)), fontsize=FS_TITLE, pad=TITLE_PAD)
        box_axes(ax)
        return im

    axes, ims = [], []
    for sec, (r, cspan) in zip(sectors, slots):
        ax = fig.add_subplot(gs[r, cspan])
        axes.append(ax)
        ims.append(heat_for_sector(ax, sec))

    # shared colorbar tight to the grid (no big gap)
    im_valid = next((im for im in ims if im is not None), None)
    if im_valid is not None:
        cbar = fig.colorbar(im_valid, cax=cax)
        cbar.set_label("IRR [%]", fontsize=FS_CBAR)
        cbar.ax.tick_params(labelsize=FS_TICK)

    # Tight margins around the whole grid
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.06, top=0.96, wspace=0.02, hspace=0.32)
    plt.show()

# Tornado chart across sectors
def Combined_Tornado_All_Sectors(data):
# Grouped bars per sector with dividers between sectors, Tol Bright colors, and numeric labels.
    df = data.get("df_mc")
    if df is None or "Sector" not in df.columns: return
    sectors = sorted(df["Sector"].dropna().unique().tolist())
    # H2 with subscript; square-bracket units
    vars_cols = [
        (r"H$_2$ price [$/kg]", TORNADO_COLORS[0]),
        (r"Carbon price [$/t]", TORNADO_COLORS[1]),
        (r"Subsidy [$/kg]",     TORNADO_COLORS[2])
    ]
    betamat = []
    for sec in sectors:
        sub = df[df["Sector"]==sec]
        row = []
        for var, _ in vars_cols:
            if {"IRR%",var}.issubset(sub.columns):
                b = _std_beta(sub[var], sub["IRR%"])
                row.append(0 if (b is None or not np.isfinite(b)) else abs(b))
            else:
                row.append(0.0)
        betamat.append(row)
    if not betamat: return
    betamat = np.array(betamat)
    x = np.arange(len(sectors), dtype=float)
    bar_w = 0.22; gap=0.04
    fig, ax = plt.subplots(figsize=(max(11, 0.8*len(sectors)+7), 6.0))
    ymax = float(np.nanmax(betamat)) if np.isfinite(betamat).any() else 1.0
    for j, (vname, col) in enumerate(vars_cols):
        offs = (j - (len(vars_cols)-1)/2.0) * (bar_w + gap)
        ax.bar(x + offs, betamat[:, j], width=bar_w, color=col, edgecolor="black", label=vname)
        for xi, b in zip(x + offs, betamat[:, j]):
            if np.isfinite(b):
                ax.text(xi, b + 0.03*ymax, f"{b:.2f}", ha="center", va="bottom", fontsize=9, color=GRAY_DARK)
    ax.set_ylim(0, ymax*1.15 if ymax>0 else 1.0)
    for i in range(len(sectors)-1):
        ax.axvline(i + 0.5, color=GRAY_LIGHT, lw=1.0, ls="-", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([_britify(_shorten(s, 14)) for s in sectors], rotation=0)
    ax.set_ylabel("|beta| on IRR")
    legend_outside_top(ax, ncol=3)
    box_axes(ax); plt.show()

# MC hist grid (all sectors), last row centered if incomplete
# Histogram grids of Monte Carlo IRR results by sector
def MC_hist_grid(data, bins=24):
    df_mc = data.get("df_mc")
    if df_mc is None or "IRR%" not in df_mc.columns or "Sector" not in df_mc.columns:
        return
    sectors = sorted(df_mc["Sector"].dropna().unique())
    n = len(sectors); cols = 3
    rows = int(np.ceil(n/cols))

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(4.8*cols, 3.8*rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.28, hspace=0.35)

    for idx, sec in enumerate(sectors):
        r = idx // cols
        if r == rows-1 and n % cols != 0:
            rcols = n % cols
            start = (cols - rcols)//2
            c = start + (idx % cols)
        else:
            c = idx % cols
        ax = fig.add_subplot(gs[r, c])
        s = df_mc[df_mc["Sector"]==sec]["IRR%"].dropna().values
        if s.size:
            ax.hist(s, bins=bins, edgecolor="black", color=TOL_BRIGHT[idx % len(TOL_BRIGHT)])

        ax.set_xlabel("IRR [%]"); ax.set_ylabel("Frequency")
        box_axes(ax)

    plt.tight_layout(); plt.show()

# F22 — Distance elasticity (no vertical bin guides by default)
# Log-log regression of flow vs distance (distance elasticity)
def F22_flow_distance_elasticity(data, show_bin_guides: bool = False):
    df = data.get("df_trade_det")
    if df is None or not {"Distance_km","Flow_kt"}.issubset(df.columns):
        return

    sub = df[(df["Flow_kt"]>0) & (df["Distance_km"]>0)].copy()
    sub["logF"] = np.log(sub["Flow_kt"])
    sub["logD"] = np.log(sub["Distance_km"])

    # OLS (in logs) slope = distance elasticity
    coef = np.polyfit(sub["logD"].to_numpy(), sub["logF"].to_numpy(), 1)  # OLS slope in logs
    b1 = coef[0]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    if "Mode" in sub.columns:
        modes = sub["Mode"].astype(str).unique().tolist()
        palette = {m: TOL_BRIGHT[i % len(TOL_BRIGHT)] for i, m in enumerate(modes)}
        for m in modes:
            s = sub[sub["Mode"]==m]
            ax.scatter(s["logD"], s["logF"], s=24, alpha=0.6, edgecolor="none",
                       color=palette[m], label=m)
        legend_outside_top(ax, ncol=min(4, len(modes)))
    else:
        ax.scatter(sub["logD"], sub["logF"], s=24, alpha=0.6, edgecolor="none", color=GRAY_MED)

    # Regression line
    x = np.linspace(sub["logD"].min(), sub["logD"].max(), 200)
    y = np.poly1d(coef)(x)
    ax.plot(x, y, color=GRAY_DARK, lw=1.2)

    if show_bin_guides:
        for km in (500, 1000, 3000, 6000):
            xv = np.log(km)
            if np.isfinite(xv) and x.min() <= xv <= x.max():
                ax.axvline(xv, color=GRAY_LIGHT, lw=1.0, ls="--", zorder=0)

    # Elasticity label placed safely inside axes
    xr = x.max() - x.min()
    x_pos = x.min() + 0.80 * xr
    y_pos = np.poly1d(coef)(x_pos)
    ymin, ymax = (min(sub["logF"]), max(sub["logF"]))
    y_pos = min(max(y_pos, ymin + 0.04*(ymax - ymin)), ymax - 0.04*(ymax - ymin))

    ax.text(x_pos, y_pos, "Elasticity ~ {:.2f}".format(b1),
            ha="left", va="bottom", fontsize=36, color=GRAY_DARK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6))

    ax.set_xlabel("ln Distance [km]")
    ax.set_ylabel("ln Flow [kt/yr]")

    box_axes(ax)
    plt.show()

# F29 — World choropleths
def _market_to_country_mapping_from_trade(data):
    df_trade = data.get("df_trade_det")
    if isinstance(df_trade, pd.DataFrame) and {"To","To_Region"}.issubset(df_trade.columns):
        m = df_trade[["To","To_Region"]].drop_duplicates()
        return m.set_index("To")["To_Region"].to_dict()
    return {}

HARDCOUNTRY = {
    "JPN_KAW": "Japan",
    "KOR_ULS": "South Korea",
    "GER_WIL": "Germany",
    "NLD_ROT": "Netherlands",
    "BEL_ANR": "Belgium",
    "CHN_SHG": "China",
    "IND_GUJ": "India",
    "UK_TEE": "United Kingdom",
    "AUS_GLD": "Australia",
    "AUS_PIL": "Australia",
    "CHL_ANT": "Chile",
    "SAU_NEOM": "Saudi Arabia",
    "OMN_DUQ": "Oman",
    "UAE_RUW": "United Arab Emirates",
    "NAM_LUD": "Namibia",
    "RSA_BOE": "South Africa",
    "MAR_SOU": "Morocco",
    "MRT_NDB": "Mauritania",
    "EGY_SUE": "Egypt",
    "KAZ_MNG": "Kazakhstan",
    "USA_HOU": "United States of America",
    "CAN_PTT": "Canada",
    "NOR_BER": "Norway",
    "UK_TEE_SUP": "United Kingdom",
}

# World map choropleths of delivered prices
def F29_price_world_map(data, highlight_markets=None):
    df = _delivered_df(data)
    if df is None or df.empty: return
    years = sorted(df["Year"].unique()); y0, y1 = years[0], years[-1]
    world = _load_world_gdf()
    trade_map = _market_to_country_mapping_from_trade(data)
    def to_country(m):
        if m in HARDCOUNTRY: return HARDCOUNTRY[m]
        if m in trade_map: return trade_map[m]
        if isinstance(m, str) and "_" in m:
            code = m.split("_")[0]
            return HARDCOUNTRY.get(code, m)
        return m

    df["GeoName"] = df["Market"].map(to_country)
    geo = df.groupby(["Year","GeoName"])["Price"].median().reset_index()

    if GEO_OK and world is not None:
        def plot_year(ax, y):
            sub = geo[geo["Year"]==y].copy()
            g = world.merge(sub, left_on("name"), right_on("GeoName"), how="left")
            vals = g["Price"].to_numpy(dtype=float)
            vmin=vmax=None
            finite = np.isfinite(vals)
            if finite.sum() > 0:
                vmin, vmax = np.nanpercentile(vals[finite], [5,95])
            g.plot(column="Price", ax=ax, legend=False, cmap="viridis",
                   vmin=vmin, vmax=vmax,
                   edgecolor="#777777", linewidth=0.3,
                   missing_kwds={"color":"lightgrey","hatch":"///","label":"No data"})
            if highlight_markets:
                names = set(pd.Series(highlight_markets).map(to_country))
                gg = g[g["name"].isin(names)]
                if len(gg):
                    gg.boundary.plot(ax=ax, color="black", linewidth=1.0)

        fig, axs = plt.subplots(1, 2, figsize=(13.6, 6.0))
        plot_year(axs[0], y0); plot_year(axs[1], y1)
        sm = mpl.cm.ScalarMappable(cmap="viridis"); sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02); cbar.set_label("Price [$/kg]")
        plt.tight_layout(); plt.show()
    else:
        pivot = geo.pivot(index="GeoName", columns="Year", values="Price").sort_index()
        fig, ax = plt.subplots(figsize=(10.8, 0.36*len(pivot)+1.6))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels(pivot.index.tolist())
        ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels(pivot.columns.astype(str))

        cbar = plt.colorbar(im, ax=ax); cbar.set_label("Price [$/kg]")
        box_axes(ax); plt.tight_layout(); plt.show()

# ========= NEW DUAL / COPY FIGURES  =========

# Dual panel: price distributions + ECDF
def Dual_F1_F3_distributions_ecdf(data, fig_dir: Path):
    df = _delivered_df(data)
    if df is None or df.empty: return
    stats = _price_stats(df)
    years = stats["Year"].tolist()
    if not years: return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.2, 5.8))
    fig.subplots_adjust(wspace=0.34)  # a bit wider gap for bigger labels

    # Left: F1-style violins
    viol = axL.violinplot([df[df["Year"] == y]["Price"].dropna().values for y in years],
                          positions=np.arange(len(years))+1, showextrema=False)
    for pc in viol['bodies']:
        pc.set_facecolor("#d9d9d9"); pc.set_edgecolor("none"); pc.set_alpha(0.95)
    axL.plot(np.arange(len(years))+1, stats["Median"], marker="o", lw=1.8, color=TOL_BRIGHT[4], label="Median")
    axL.fill_between(np.arange(len(years))+1, stats["P25"], stats["P75"], alpha=0.25, color=TOL_BRIGHT[4], label="IQR (P25-P75)")
    axL.plot(np.arange(len(years))+1, stats["P10"], ls="--", color=GREEN_STRONG, label="P10")
    axL.plot(np.arange(len(years))+1, stats["P90"], ls="--", color=RED_STRONG,   label="P90")
    axL.set_xticks(np.arange(len(years))+1); axL.set_xticklabels(years)
    axL.set_xlabel("Year"); axL.set_ylabel("Price [$/kg]")
    box_axes(axL); legend_outside_top(axL, ncol=4, y=1.02, top_pad=0.94)

    # Right: F3-style ECDF
    years_sorted = sorted(df["Year"].unique())
    for i, y in enumerate(years_sorted):
        s = df[df["Year"] == y]["Price"].dropna().sort_values().values
        if s.size:
            F = np.arange(1, len(s)+1)/len(s)
            axR.step(
                s, F, where="post", label=str(y),
                color=YEAR_COLORS[i % len(YEAR_COLORS)]
            )
    axR.set_xlabel("Price [$/kg]"); axR.set_ylabel("Cumulative share [%]")
    _percent_formatter01(axR)
    box_axes(axR); legend_outside_top(axR, ncol=4, y=1.02, top_pad=0.94)

    _save_named(fig, fig_dir, "dual_F1_F3_distributions_ecdf")

def Copy_F10_mode_distance_combo(data, fig_dir: Path):
    df = data.get("df_trade_det")
    if df is None or "Mode" not in df.columns: return
    dist_col = None
    for c in ("Distance_km","Dist_km","distance_km"):
        if c in df.columns: dist_col = c; break
    if dist_col is None:
        return
    sub = df.copy()
    bins = [0, 500, 1000, 3000, 6000, np.inf]
    labels = ["<0.5k", "0.5-1k", "1-3k", "3-6k", ">6k"]
    sub["bin"] = pd.cut(sub[dist_col], bins=bins, labels=labels, include_lowest=True)
    if "Unit_Cost" not in sub.columns:
        sub["Unit_Cost"] = np.nan
    med = sub.groupby("bin", observed=True)["Unit_Cost"].median().reindex(labels)

    fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.6))
    modes = sorted(df["Mode"].dropna().unique().tolist())
    palette = {m: TOL_BRIGHT[i%len(TOL_BRIGHT)] for i,m in enumerate(modes)}
    for m in modes:
        s = df[df["Mode"]==m].copy()
        axs[0].scatter(s[dist_col], s.get("Unit_Cost", np.nan), s=24, alpha=0.65, edgecolor="none",
                       color=palette[m], label=str(m))
    axs[0].set_xlabel("Distance [km]"); axs[0].set_ylabel("Unit transport cost [$/unit]")
    if np.isfinite(df[dist_col]).any():
        xmax = float(np.nanmax(df[dist_col].values))
        axs[0].set_xlim(0, xmax*1.05)
    box_axes(axs[0]); legend_outside_top(axs[0], ncol=min(4,len(modes)), y=1.10, top_pad=0.90)

    axs[1].plot(np.arange(len(labels)), med.values, "-o", color=GRAY_DARK)
    axs[1].set_xticks(np.arange(len(labels))); axs[1].set_xticklabels(labels)
    axs[1].set_xlabel("Distance bin [km]"); axs[1].set_ylabel("Median unit cost [$/unit]")
    box_axes(axs[1])
    plt.tight_layout()
    _save_named(fig, fig_dir, "dual_F10_mode_distance_combo")

# Dual panel: structure vs elasticity
def Dual_F11_F27_structure_vs_elasticity(data, fig_dir: Path):
    # Left: F7 (market structure vs dispersion)
    DIV_GREY = "#e6e6e6"  # unified light grey for dotted leaders/separators

    df_trade = data.get("df_trade_det"); df_deliv = _delivered_df(data)
    if df_trade is None or df_deliv is None: return
    flow_col = "Flow_kt" if "Flow_kt" in df_trade.columns else None
    if flow_col is None: return

    hh = []
    for y, sub in df_trade.groupby("Year"):
        s = sub.groupby("From")[flow_col].sum()
        if s.sum() == 0: continue
        share = (s/s.sum())**2  # Herfindahl components
        hh.append((y, float(share.sum()*10000)))
    if not hh: return
    years_h, hhis = zip(*sorted(hh))
    stats = _price_stats(df_deliv)
    disp = stats.set_index("Year")["Band"].reindex(years_h)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.0, 5.8))
    fig.subplots_adjust(wspace=0.34)  # a bit wider gap for bigger labels

    axL.plot(years_h, hhis, "-o", color=BLUE_YELLOW["blue"],  label="Export HHI")
    axL.set_ylabel("HHI (index)"); axL.set_xlabel("Year"); box_axes(axL)
    axL2 = axL.twinx()
    axL2.plot(years_h, disp.values, "--s", color=BLUE_YELLOW["yellow"], label="P90-P10 price band")
    axL2.set_ylabel("Price dispersion [$/kg]"); box_axes(axL2)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axL2.get_legend_handles_labels()
    leg = axL.legend(h1 + h2, l1 + l2,  # <— correct concatenation
                     loc="lower center", bbox_to_anchor=(0.5, 1.02),
                     ncol=2, frameon=True, fancybox=False, edgecolor="black")
    leg.get_frame().set_linewidth(1.1)
    axL.figure.subplots_adjust(top=0.94)

    # Right: F22 (distance elasticity) — bin guides off
    df = df_trade.copy()
    sub = df[(df["Flow_kt"]>0) & (df["Distance_km"]>0)].copy()
    sub["logF"] = np.log(sub["Flow_kt"]); sub["logD"] = np.log(sub["Distance_km"])
    coef = np.polyfit(sub["logD"].to_numpy(), sub["logF"].to_numpy(), 1)  # OLS slope in logs; b1 = coef[0]
    if "Mode" in sub.columns:
        modes = sub["Mode"].astype(str).unique().tolist()
        palette = {m: TOL_BRIGHT[i % len(TOL_BRIGHT)] for i, m in enumerate(modes)}
        for m in modes:
            s = sub[sub["Mode"]==m]
            axR.scatter(s["logD"], s["logF"], s=24, alpha=0.6, edgecolor="none", color=palette[m], label=m)
        legend_outside_top(axR, ncol=min(4, len(modes)), y=1.02, top_pad=0.94)  # tightened
    else:
        axR.scatter(sub["logD"], sub["logF"], s=24, alpha=0.6, edgecolor="none", color=GRAY_MED)
        # Regression line
    x = np.linspace(sub["logD"].min(), sub["logD"].max(), 200)
    y = np.poly1d(coef)(x)
    axR.plot(x, y, color=GRAY_DARK, lw=1.2)

    # Leader from the centre of the regression line down to a label
    xmin, xmax = x.min(), x.max()
    xr = xmax - xmin
    x_mid = xmin + 0.50 * xr              # centre of the line
    y_mid = np.poly1d(coef)(x_mid)

    ymin, ymax = (min(sub["logF"]), max(sub["logF"]))
    y_span = ymax - ymin
    y_lab  = max(ymin + 0.08*y_span, y_mid - 0.18*y_span)  # place label below the line midpoint

    # dotted grey leader (same grey as our separators)
    axR.annotate(
        "", xy=(x_mid, y_mid), xytext=(x_mid, y_lab),
        arrowprops=dict(arrowstyle="-", linestyle=":", color=DIV_GREY, lw=1.4),
        annotation_clip=False,
    )

    # elasticity label with transparent white background
    txt = axR.text(
        x_mid, y_lab, "Elasticity = {:.2f}".format(b1),
        ha="center", va="top",
        fontsize=14, color=GRAY_DARK, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
        clip_on=False,
    )
    txt.set_fontsize(14)

    axR.set_xlabel("ln Distance [km]"); axR.set_ylabel("ln Flow [kt/yr]"); box_axes(axR)

    _save_named(fig, fig_dir, "dual_F11_F27_structure_vs_elasticity")

# Dual panel: tornado vs Monte Carlo IRR
def Dual_F22_F26_tornado_vs_mcIRR(data, fig_dir: Path):
    # Left: Combined Tornado (|beta| by sector)
    df = data.get("df_mc")
    if df is None or "Sector" not in df.columns or "IRR%" not in df.columns:
        return

    sectors = sorted(df["Sector"].dropna().unique().tolist())

    # Use data keys for calculation; pretty labels (with H₂ + escaped $) for legend
    vars_spec = [
        ("H2 Price",     r"H$_2$ price [\$/kg]", TORNADO_COLORS[0]),
        ("Carbon $/t",   r"Carbon price [\$/t]", TORNADO_COLORS[1]),
        ("Subsidy $/kg", r"Subsidy [\$/kg]",     TORNADO_COLORS[2]),
    ]

    betamat = []
    for sec in sectors:
        sub = df[df["Sector"] == sec]
        row = []
        for key, _label, _col in vars_spec:
            if {"IRR%", key}.issubset(sub.columns):
                x = sub[key].to_numpy(dtype=float)
                y = sub["IRR%"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 3 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
                    xs = (x[mask] - x[mask].mean()) / x[mask].std()  # standardise inputs
                    ys = (y[mask] - y[mask].mean()) / y[mask].std()  # standardise outputs
                    b = np.polyfit(xs, ys, 1)[0]
                    row.append(abs(float(b)))
                else:
                    row.append(0.0)
            else:
                row.append(0.0)
        betamat.append(row)
    betamat = np.array(betamat)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.0, 6.0))
    fig.subplots_adjust(wspace=0.34, bottom=0.14, top=0.94)

    # Bars
    x = np.arange(len(sectors), dtype=float)
    bar_w = 0.22; gap = 0.04
    ymax = float(np.nanmax(betamat)) if np.isfinite(betamat).any() else 1.0
    for j, (_key, label, col) in enumerate(vars_spec):
        offs = (j - (len(vars_spec) - 1) / 2.0) * (bar_w + gap)
        axL.bar(x + offs, betamat[:, j], width=bar_w, color=col,
                edgecolor="black", label=label)
    axL.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
    axL.set_xticks(x)
    axL.set_xticklabels([_britify(_shorten(s, 14)) for s in sectors], rotation=0)
    axL.set_xlabel("Sector")
    axL.set_ylabel("|beta| on IRR")
    legend_outside_top(axL, ncol=3, y=1.02, top_pad=0.94)  # tightened

    # Right: MC IRR distributions (single-panel violins by sector)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    VIOLIN_FACE = "#c9d7e5"

    data_mc = [df[df["Sector"] == sec]["IRR%"].dropna().to_numpy() for sec in sectors]
    v = axR.violinplot(data_mc, positions=np.arange(len(sectors)) + 1, showextrema=False)
    for pc in v['bodies']:
        pc.set_facecolor(VIOLIN_FACE)
        pc.set_edgecolor("black")
        pc.set_alpha(0.9)

    med = [np.nanmedian(x) if len(x) else np.nan for x in data_mc]
    axR.plot(np.arange(len(sectors)) + 1, med, "o-",
             color=TOL_BRIGHT[4], lw=1.4, label="Median")

    axR.set_xticks(np.arange(len(sectors)) + 1)
    axR.set_xticklabels([_britify(_shorten(s, 14)) for s in sectors], rotation=0)
    axL.set_xlabel("Sector")
    axR.set_ylabel("IRR [%]")
    box_axes(axR)

    handles = [
        Patch(facecolor=VIOLIN_FACE, edgecolor="black", alpha=0.9, label="IRR distribution"),
        Line2D([], [], color=TOL_BRIGHT[4], marker="o", linestyle="-", linewidth=1.4, label="Median"),
    ]
    leg = axR.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                     ncol=2, frameon=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    axR.figure.subplots_adjust(top=0.94)

    _save_named(fig, fig_dir, "dual_F22_F26_tornado_vs_mcIRR")

# ============================== Main ================================= #

# Entrypoint: load data, and produce figures
def main():
    use_cambridge_style(usetex=False)
    data, _ = load_run()
    run_dir = Path(data["_dir"])
    fig_dir = run_dir / "figures"
    autosave_show(fig_dir)
    print("Loaded from:", run_dir.resolve())
    print("Saving figures to:", fig_dir.resolve())
    print_loaded_summary(data)

    # Dual: Figure 1 + Figure 3 — distributions + ECDF
    Dual_F1_F3_distributions_ecdf(data, fig_dir)

    # Dual: Figure 11 + Figure 27 — structure vs elasticity
    Dual_F11_F27_structure_vs_elasticity(data, fig_dir)

    # Dual: Figure 22 + Figure 26 — tornado vs MC IRR
    Dual_F22_F26_tornado_vs_mcIRR(data, fig_dir)

    # Figure 5 — exemplar nodes by decile (cohorted) multi-year bars
    F4_exemplars_all_years_clean(data)

    # Figure 21 — two-way IRR heatmaps by sector (5-up panel)
    Combined_Heatmaps_All_Sectors(data)

if __name__ == "__main__":
    main()
