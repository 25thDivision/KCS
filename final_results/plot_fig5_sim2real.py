"""
Figure 5 — Sim-to-real gap (flagship figure)

Each point: (decoder × distance × platform × noise_profile)
  X: Simulation ECR (%), averaged over 3 error-rate p values, max over epochs
  Y: Hardware ECR (%) = 100 × (1 − LER_ML / LER_NoCorr) per same noise profile
  Color  = decoder architecture (8)
  Shape  = platform (3 hardware families)
  Size   = distance (d=3 small, d=5 large)
  Panels = 3 noise profiles (realistic / heavy / extreme)

Reference lines:
  y = x   (perfect sim-to-real transfer)
  y = 0   (no improvement over No Correction)

Dropped by design: d=7 (no hardware counterpart), heavy-hex d>3 (code is d=3 only).
NaN-safe: missing configs (e.g. GraphTransformer d=5) drop out silently and are
listed in the console summary.

Usage:
    $ python plot_fig5_sim2real.py
    Expects CSV files in DATA_DIR. Writes fig5_sim2real_scatter.{png,pdf}.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ====================== config ======================
DATA_DIR = Path('.')                       # <-- adjust if CSVs are elsewhere
OUT_STEM = 'fig5_sim2real_scatter'
HERON_MODE = 'pooled'                       # 'pooled' (Aachen+Pittsburgh combined) | 'separate'
INCLUDE_BOSTON = True                       # Boston = 1 run; still informative as Heron point
PLOT_MODE = 'box'
# 'mean':     one averaged point per (decoder × d × noise × platform)  — cleanest
# 'raw':      one small point per HW run (no averaging)                — honest spread
# 'raw+mean': small raw dots + large mean marker on top                — both at once
# 'box':      box (Q1–Q3) + whiskers (min/max) + median tick           — variance summary

# Each row in HW CSVs contains `Stim_Error_Rate` = the Stim depolarizing p the
# decoder was trained with. We match this to sim's `Error_Rate(p)` for a like-for-like
# sim↔hw comparison. Picking p=0.005 as canonical: lowest training p, closest to the
# actual hardware physical error regime. Other training p values (0.01, 0.05) produce
# pathological weights for graph-conv architectures and distort the picture if averaged.
CANONICAL_P = 0.005

MODELS = ['APPNP','CNN','GAT','GCN','GCNII','GNN','GraphMamba','GraphTransformer']

# Color palette — group visually by architecture family
COLORS = {
    'GraphTransformer': '#d62728',   # red      — Attention (hero)
    'GraphMamba':       '#ff7f0e',   # orange   — SSM / Attention-like (hero)
    'GAT':              '#bcbd22',   # olive    — attention w/ conv bias
    'CNN':              '#2ca02c',   # green    — baseline
    'GCN':              '#1f77b4',   # blue     — graph conv
    'GCNII':            '#17becf',   # cyan     — deep GCN
    'APPNP':            '#9467bd',   # purple   — personalized PageRank
    'GNN':              '#8c564b',   # brown    — vanilla MPNN
}

PLATFORM_MARKER = {
    'Forte-1':   '^',   # triangle-up
    'Heron R3':  'o',   # circle
    'Nighthawk': 's',   # square
}

# Code names in sim CSV ↔ hardware platform
PLATFORM_CODE = {
    'Forte-1':   'color_code',
    'Heron R3':  'heavyhex_surface_code',
    'Nighthawk': 'surface_code',
}

# Noise profile mapping (hardware CSV has 'realistic/' prefix, sim CSV doesn't)
NOISE_LABEL = {
    'dp0.001_mf0.01_rf0.01_gd0.008': 'realistic',
    'dp0.005_mf0.02_rf0.02_gd0.015': 'heavy',
    'dp0.01_mf0.05_rf0.05_gd0.01':   'extreme',
}
NOISE_ORDER = list(NOISE_LABEL.keys())


# ====================== 1. Load CSVs ======================
sim        = pd.read_csv(DATA_DIR / 'conbined_stim.csv')
forte1     = pd.read_csv(DATA_DIR / 'combined_forte1.csv')
miami      = pd.read_csv(DATA_DIR / 'combined_ibm_miami.csv')
boston     = pd.read_csv(DATA_DIR / 'combined_ibm_boston.csv')
aachen     = pd.read_csv(DATA_DIR / 'combined_ibm_aachen.csv')
pittsburgh = pd.read_csv(DATA_DIR / 'combined_ibm_pittsburgh.csv')


# ====================== 2. Simulation ECR (X axis) ======================
# Keep d ≤ 5 (d=7 out of scope). Heavy-hex d>3 rows won't exist anyway.
sim = sim[sim['Distance'] <= 5].copy()

# Multiple rows per config correspond to training epochs — take the best (max ECR).
sim_best = (sim.groupby(['code', 'Distance', 'noise', 'Error_Rate(p)', 'model'])
               ['Best_ECR(%)'].max().reset_index())

# Use only the canonical training p (matched to HW Stim_Error_Rate later).
sim_x = (sim_best[sim_best['Error_Rate(p)'] == CANONICAL_P]
         [['code', 'Distance', 'noise', 'model', 'Best_ECR(%)']]
         .rename(columns={'Best_ECR(%)': 'ECR_sim'}))


# ====================== 3. Hardware ECR (Y axis) ======================
def compute_hw_ecr(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """Return per-run raw ECR for the canonical training-p only.
    ECR = 100·(1 − LER/LER_NoCorr). NoCorr baseline = mean of NoCorr runs per distance."""
    base = (df[df['Model'] == 'No_Correction']
            .groupby('Distance')['Logical_Error_Rate'].mean())

    # Filter to canonical training p to match the sim weight
    ml = df[df['Weight_Noise'].notna()
            & df['Model'].isin(MODELS)
            & (df['Stim_Error_Rate'] == CANONICAL_P)].copy()
    ml['LER_nocorr'] = ml['Distance'].map(base)
    ml['ECR_hw']     = 100 * (1 - ml['Logical_Error_Rate'] / ml['LER_nocorr'])
    ml['platform']   = platform
    ml['noise']      = ml['Weight_Noise'].str.replace('realistic/', '', regex=False)
    ml['code']       = PLATFORM_CODE[platform]
    ml = ml.rename(columns={'Model': 'model',
                            'Logical_Error_Rate': 'LER_hw'})
    return ml[['model', 'Distance', 'noise', 'Run',
               'ECR_hw', 'LER_hw', 'LER_nocorr',
               'platform', 'code']]


hw_frames = [compute_hw_ecr(forte1, 'Forte-1')]

if HERON_MODE == 'pooled':
    heron_parts = [aachen, pittsburgh]
    if INCLUDE_BOSTON:
        heron_parts.append(boston)
    heron = pd.concat(heron_parts, ignore_index=True)
    hw_frames.append(compute_hw_ecr(heron, 'Heron R3'))
else:
    # Keep each backend as its own point — labeled identically as Heron R3 for shape
    # but differentiated in the debug print
    for name, dfi in [('aachen', aachen), ('pittsburgh', pittsburgh)] + \
                     ([('boston', boston)] if INCLUDE_BOSTON else []):
        frame = compute_hw_ecr(dfi, 'Heron R3')
        frame['_backend'] = name
        hw_frames.append(frame)

hw_frames.append(compute_hw_ecr(miami, 'Nighthawk'))
hw = pd.concat(hw_frames, ignore_index=True)


# ====================== 4. Merge sim ↔ hw ======================
# hw is now per-run (long format). Merge sim values onto each run row.
merged_raw = hw.merge(sim_x, on=['code', 'Distance', 'noise', 'model'], how='left')
full_raw   = merged_raw.dropna(subset=['ECR_sim', 'ECR_hw']).copy()

# Also compute per-condition means for optional overlay / 'mean' mode
mean_cols = ['model', 'Distance', 'noise', 'platform', 'code', 'ECR_sim']
full_mean = (full_raw.groupby(mean_cols)['ECR_hw']
                    .agg(['mean', 'std', 'count']).reset_index()
                    .rename(columns={'mean': 'ECR_hw', 'count': 'n_runs'}))

print(f"\n{'='*66}\nFigure 5 data summary\n{'='*66}")
print(f"Raw HW run rows           : {len(merged_raw)}")
print(f"Plotted raw points        : {len(full_raw)}")
print(f"Condition-mean points     : {len(full_mean)}")
print(f"\nRaw points per platform × distance:")
print(full_raw.groupby(['platform', 'Distance']).size().unstack(fill_value=0))
print(f"\nRuns per condition (raw mode sanity check):")
print(full_mean.groupby(['platform', 'Distance'])['n_runs'].agg(['min', 'max']))


# ====================== 5. Plot ======================
Y_CLIP = (-120, 40)                         # most decoders sit in [-100, 30]; clip deeper failures
GOOD_COLOR = '#e8f4ea'                      # pale green for y > 0 region
BAD_COLOR  = '#fbeceb'                      # pale red for y < 0 region

# add x-jitter so stacked runs don't perfectly overlap
RNG = np.random.default_rng(0)
JITTER = 0.6                                # percent on x-axis

# Marker size per (d, role)
SIZE_RAW_D3,   SIZE_RAW_D5   = 18, 42
SIZE_MEAN_D3,  SIZE_MEAN_D5  = 95, 180


def draw_points(ax, df, kind, alpha):
    """kind in {'raw', 'mean'}. alpha overrides."""
    size3 = SIZE_RAW_D3   if kind == 'raw' else SIZE_MEAN_D3
    size5 = SIZE_RAW_D5   if kind == 'raw' else SIZE_MEAN_D5
    edge  = 0.25 if kind == 'raw' else 0.9
    for _, r in df.iterrows():
        is_d5 = r['Distance'] == 5
        y = max(r['ECR_hw'], Y_CLIP[0] + 3)
        clipped = r['ECR_hw'] < Y_CLIP[0] + 3
        x = r['ECR_sim'] + (RNG.uniform(-JITTER, JITTER) if kind == 'raw' else 0)
        ax.scatter(x, y,
                   marker=PLATFORM_MARKER[r['platform']],
                   color=COLORS[r['model']],
                   s=size5 if is_d5 else size3,
                   alpha=alpha,
                   edgecolors='black', linewidth=edge, zorder=3 if kind == 'raw' else 4)
        if clipped and kind == 'mean':
            ax.annotate('', xy=(x, Y_CLIP[0] + 1.5),
                        xytext=(x, Y_CLIP[0] + 7),
                        arrowprops=dict(arrowstyle='-|>', color=COLORS[r['model']],
                                        lw=1.2, alpha=0.8), zorder=5)


# Box-whisker mode: per (decoder × d × noise × platform), draw:
#   - box from Q1 to Q3 filled with decoder color
#   - median tick (horizontal black line inside box)
#   - whiskers (thin black vertical lines) from min up to Q1 and Q3 up to max
#   - platforms are separated by small x-offset (so same decoder across 3 platforms sits side-by-side)
#   - box width encodes distance (d=3 narrow, d=5 wide)
from matplotlib.patches import Rectangle

BOX_W_D3 = 1.6
BOX_W_D5 = 2.8
PLATFORM_X_OFFSET = {'Forte-1': -2.6, 'Heron R3': 0.0, 'Nighthawk': 2.6}


def draw_box(ax, df):
    for _, r in df.iterrows():
        vals = np.asarray(r['runs'])
        is_d5 = r['Distance'] == 5
        w = BOX_W_D5 if is_d5 else BOX_W_D3
        x0 = r['ECR_sim'] + PLATFORM_X_OFFSET[r['platform']]
        color = COLORS[r['model']]
        # Values clipped inside visible y-range for box drawing
        vals_vis = np.clip(vals, Y_CLIP[0] + 3, Y_CLIP[1])
        any_clipped = (vals < Y_CLIP[0] + 3).any()

        if len(vals) == 1:
            # Single run — show as a filled marker in platform shape
            y = vals_vis[0]
            ax.scatter(x0, y,
                       marker=PLATFORM_MARKER[r['platform']],
                       color=color,
                       s=(SIZE_MEAN_D5 if is_d5 else SIZE_MEAN_D3) * 0.8,
                       alpha=0.92,
                       edgecolors='black', linewidth=0.9, zorder=4)
            if any_clipped:
                ax.annotate('', xy=(x0, Y_CLIP[0] + 1.5),
                            xytext=(x0, Y_CLIP[0] + 7),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                            lw=1.2, alpha=0.8), zorder=5)
            continue

        q1, med, q3 = np.percentile(vals_vis, [25, 50, 75])
        lo, hi = vals_vis.min(), vals_vis.max()

        # Whiskers
        ax.plot([x0, x0], [lo, q1], color='black', lw=0.9, alpha=0.85, zorder=3)
        ax.plot([x0, x0], [q3, hi], color='black', lw=0.9, alpha=0.85, zorder=3)
        # Whisker caps (small horizontals)
        cap_w = w * 0.4
        ax.plot([x0 - cap_w, x0 + cap_w], [lo, lo],
                color='black', lw=0.9, alpha=0.85, zorder=3)
        ax.plot([x0 - cap_w, x0 + cap_w], [hi, hi],
                color='black', lw=0.9, alpha=0.85, zorder=3)
        # Box (filled Q1–Q3 rectangle)
        ax.add_patch(Rectangle((x0 - w/2, q1), w, q3 - q1,
                               facecolor=color, edgecolor='black',
                               linewidth=0.8, alpha=0.85, zorder=4))
        # Median tick — white line overlaid on dark so visible on any color
        ax.plot([x0 - w/2, x0 + w/2], [med, med],
                color='white', lw=1.8, alpha=1.0, zorder=5, solid_capstyle='butt')
        ax.plot([x0 - w/2, x0 + w/2], [med, med],
                color='black', lw=0.5, alpha=0.8, zorder=6, solid_capstyle='butt')

        if any_clipped:
            ax.annotate('', xy=(x0, Y_CLIP[0] + 1.5),
                        xytext=(x0, Y_CLIP[0] + 7),
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        lw=1.2, alpha=0.8), zorder=5)


fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)

# Build per-cluster run list once (for box mode)
box_df = (full_raw.groupby(['model', 'Distance', 'noise', 'platform', 'ECR_sim'])
                  ['ECR_hw'].apply(list).reset_index()
                  .rename(columns={'ECR_hw': 'runs'}))

for ax, nk in zip(axes, NOISE_ORDER):
    # --- region shading ---
    ax.axhspan(0, Y_CLIP[1],     facecolor=GOOD_COLOR, alpha=0.55, zorder=0)
    ax.axhspan(Y_CLIP[0], 0,     facecolor=BAD_COLOR,  alpha=0.35, zorder=0)

    # --- reference lines ---
    ax.plot([-20, 100], [-20, 100], ls='--', color='#444', lw=1.3,
            alpha=0.75, zorder=1)
    ax.axhline(0, color='black', lw=1.0, alpha=0.65, zorder=1)

    # Annotate y = x
    ax.annotate('y = x', xy=(92, 92), xytext=(83, 35),
                fontsize=9, color='#444', style='italic',
                arrowprops=dict(arrowstyle='->', color='#444', lw=0.7, alpha=0.6))

    # --- scatter ---
    if PLOT_MODE in ('raw', 'raw+mean'):
        draw_points(ax, full_raw[full_raw['noise'] == nk],
                    kind='raw', alpha=0.42)
    if PLOT_MODE in ('mean', 'raw+mean'):
        draw_points(ax, full_mean[full_mean['noise'] == nk],
                    kind='mean', alpha=0.92)
    if PLOT_MODE == 'box':
        draw_box(ax, box_df[box_df['noise'] == nk])

    # Count of clipped (based on mean in mean/mixed; raw min in raw/box mode)
    if PLOT_MODE in ('raw', 'box'):
        sub = full_raw[full_raw['noise'] == nk]
    else:
        sub = full_mean[full_mean['noise'] == nk]
    n_clipped = (sub['ECR_hw'] < Y_CLIP[0] + 3).sum()

    title = f"noise profile: {NOISE_LABEL[nk]}"
    if n_clipped > 0:
        title += f"   ({n_clipped} below axis)"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Simulation ECR (%)', fontsize=11)
    ax.grid(True, alpha=0.2, zorder=1)
    ax.set_xlim(-3, 100)
    ax.set_ylim(Y_CLIP)

    # Region labels (only leftmost panel)
    if ax is axes[0]:
        ax.text(2, 25, 'improvement\nover NoCorr',
                fontsize=8.5, color='#2a7a36', alpha=0.9, ha='left', va='center',
                style='italic')
        ax.text(2, -55, 'worse than\nNoCorr',
                fontsize=8.5, color='#9b2a24', alpha=0.9, ha='left', va='center',
                style='italic')

axes[0].set_ylabel(
    r'Hardware ECR (%) $= 100 \cdot (1 - \mathrm{LER}_{\mathrm{ML}}/\mathrm{LER}_{\mathrm{NoCorr}})$',
    fontsize=11)

# ---- Custom legend (outside right) ----
dec_handles = [Line2D([0], [0], marker='o', linestyle='', color='w',
                      markerfacecolor=COLORS[m], markeredgecolor='black',
                      markersize=9, label=m) for m in MODELS]
plat_handles = [Line2D([0], [0], marker=PLATFORM_MARKER[p], linestyle='',
                       color='lightgray', markeredgecolor='black',
                       markersize=10, label=p) for p in PLATFORM_MARKER]
dist_handles = [Line2D([0], [0], marker='o', linestyle='', color='lightgray',
                       markeredgecolor='black', markersize=sz, label=f'd = {d}')
                for sz, d in [(7, 3), (11, 5)]]

leg1 = fig.legend(handles=dec_handles,  title='Decoder',
                  loc='center left', bbox_to_anchor=(1.005, 0.68),
                  fontsize=9, frameon=False, title_fontsize=10)
leg2 = fig.legend(handles=plat_handles, title='Platform',
                  loc='center left', bbox_to_anchor=(1.005, 0.36),
                  fontsize=9, frameon=False, title_fontsize=10)
leg3 = fig.legend(handles=dist_handles, title='Distance',
                  loc='center left', bbox_to_anchor=(1.005, 0.16),
                  fontsize=9, frameon=False, title_fontsize=10)

fig.suptitle('Sim-to-real gap: simulation ECR vs hardware ECR',
             fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0, 0.98, 1.0])

for ext in ('png', 'pdf'):
    out = DATA_DIR / f'{OUT_STEM}_{PLOT_MODE}.{ext}'
    plt.savefig(out, dpi=220, bbox_inches='tight')
    print(f"Saved: {out}")

plt.close()
