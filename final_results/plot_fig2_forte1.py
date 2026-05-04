"""
Figure 2 — IonQ Forte-1 deep dive

Two-row panel structure:
  Row (a): Standalone decoders — NoCorr · MWPM_Restriction · 8 ML decoders
  Row (b): Hybrid MWPM+ML — MWPM_Restriction · 8 hybrid decoders
With two columns: d=3 (left), d=5 (right).

For each (model, distance), bars show:
  - mean LER across 5 runs at canonical training p (0.005)
  - error bar = ±1 standard deviation across runs
  - bar color = decoder architecture (consistent with Fig 5)
  - colored band overlay = realistic / heavy / extreme noise profile (3 bars per model)

Reference horizontal lines: NoCorr mean (red dashed), MWPM_Restriction mean (gray dashed).
This figure complements Fig 5 by showing absolute LER values, MWPM comparison,
and the hybrid-decoder null result that Fig 5's ECR-ratio framing hides.

Usage:
    $ python plot_fig2_forte1.py
"""
import sys
sys.path.insert(0, '.')
from fig_common import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT_STEM = 'fig2_forte1'

# Three bars per model, one per noise profile, side-by-side
NOISE_HATCH = {
    'dp0.001_mf0.01_rf0.01_gd0.008': '',     # realistic — solid
    'dp0.005_mf0.02_rf0.02_gd0.015': '//',   # heavy
    'dp0.01_mf0.05_rf0.05_gd0.01':   'xx',   # extreme
}

# ---------- Load Forte-1 only ----------
df = pd.read_csv(DATA_DIR / F_FORTE1)
df = df[df['Stim_Error_Rate'] == CANONICAL_P].copy()

# Strip 'realistic/' prefix from Weight_Noise
df['noise'] = df['Weight_Noise'].str.replace('realistic/', '', regex=False)

# Baselines: NoCorr / MWPM_Restriction (no Weight_Noise)
base_df = pd.read_csv(DATA_DIR / F_FORTE1)
base_df = base_df[base_df['Weight_Noise'].isna()
                  & base_df['Model'].isin(['No_Correction', 'MWPM_Restriction'])]
base_stats = (base_df.groupby(['Model', 'Distance'])['Logical_Error_Rate']
                     .agg(['mean', 'std']).reset_index())

print("Baselines (5 runs each):")
print(base_stats.to_string(index=False))


# ---------- Aggregate per (Model, Distance, noise) ----------
agg = (df[df['Weight_Noise'].notna()]
       .groupby(['Model', 'Distance', 'noise'])['Logical_Error_Rate']
       .agg(['mean', 'std', 'count']).reset_index())
agg['is_hybrid'] = agg['Model'].str.startswith('MWPM+')
agg['model']     = agg['Model'].str.replace('MWPM+', '', regex=False)


# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex='col')

bar_w = 0.26    # per-noise sub-bar width
group_gap = 1.0 # space between decoder groups (in 'model index' units)

for col, dist in enumerate([3, 5]):
    for row, is_hybrid in enumerate([False, True]):
        ax = axes[row, col]

        sub = agg[(agg['Distance'] == dist) & (agg['is_hybrid'] == is_hybrid)]

        # Bar positions
        for mi, m in enumerate(MODELS):
            for ni, nk in enumerate(NOISE_ORDER):
                row_data = sub[(sub['model'] == m) & (sub['noise'] == nk)]
                if len(row_data) == 0:
                    continue
                r = row_data.iloc[0]
                x = mi * group_gap + (ni - 1) * bar_w
                ax.bar(x, r['mean'], width=bar_w * 0.94,
                       color=COLORS[m], edgecolor='black', linewidth=0.5,
                       hatch=NOISE_HATCH[nk], alpha=0.9, zorder=3)
                ax.errorbar(x, r['mean'], yerr=r['std'], fmt='none',
                            ecolor='black', lw=0.8, capsize=2, zorder=4)

        # Reference lines
        nc = base_stats[(base_stats['Model'] == 'No_Correction')
                        & (base_stats['Distance'] == dist)].iloc[0]
        mw = base_stats[(base_stats['Model'] == 'MWPM_Restriction')
                        & (base_stats['Distance'] == dist)].iloc[0]
        ax.axhline(nc['mean'], color='#c00', ls='--', lw=1.3, alpha=0.85, zorder=2,
                   label=f"NoCorr ({nc['mean']:.3f}±{nc['std']:.3f})")
        ax.axhline(mw['mean'], color='#444', ls='--', lw=1.3, alpha=0.85, zorder=2,
                   label=f"MWPM ({mw['mean']:.3f}±{mw['std']:.3f})")
        # NoCorr ± std band
        ax.axhspan(nc['mean'] - nc['std'], nc['mean'] + nc['std'],
                   color='#c00', alpha=0.06, zorder=1)
        ax.axhspan(mw['mean'] - mw['std'], mw['mean'] + mw['std'],
                   color='#444', alpha=0.06, zorder=1)

        # Layout
        ax.set_xticks([mi * group_gap for mi in range(len(MODELS))])
        ax.set_xticklabels(MODELS, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Logical Error Rate', fontsize=10)
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)

        kind = 'Hybrid MWPM+ML' if is_hybrid else 'Standalone ML'
        ax.set_title(f"({chr(ord('a') + row*2 + col)}) {kind}, d = {dist}",
                     fontsize=11, loc='left')

        ax.set_ylim(0, 0.30 if is_hybrid else 1.0)


# ---------- Bottom legend (noise hatch) ----------
hatch_handles = [Patch(facecolor='lightgray', edgecolor='black',
                       hatch=NOISE_HATCH[k], label=NOISE_LABEL[k])
                 for k in NOISE_ORDER]
fig.legend(handles=hatch_handles, title='noise profile',
           loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=3, fontsize=10, frameon=False, title_fontsize=10)

fig.suptitle('IonQ Forte-1 — color code, hardware LER per decoder '
             f'(p = {CANONICAL_P}, 5 runs)', fontsize=13, y=1.0)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

for ext in ('png', 'pdf'):
    out = DATA_DIR / f'{OUT_STEM}.{ext}'
    plt.savefig(out, dpi=220, bbox_inches='tight')
    print(f"Saved: {out}")
plt.close()
