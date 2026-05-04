"""
Figure 5 — Sim-to-real gap (flagship)

Each cluster: (decoder × distance × platform × noise_profile)
  X: simulation ECR (%) at canonical training p
  Y: hardware ECR (%) = 100·(1 − LER_ML / LER_NoCorr) per run
  Box  = Q1–Q3 of the runs in that cluster
  Whiskers = min and max
  Median = white line on box
  Color = decoder architecture
  X-offset within an x-cluster encodes platform
  Box width encodes distance (d=3 narrow, d=5 wide)

Reference lines:
  y = x  (perfect sim-to-real transfer)
  y = 0  (no improvement over No Correction)

Drops by design:
  - d=7 (no hardware counterpart)
  - heavy-hex d>3 (code is d=3 only)
  - hybrid MWPM+ML rows (covered in Fig 2)

Usage:
    $ python plot_fig5_sim2real.py
"""
import re
import sys
sys.path.insert(0, '.')
from fig_common import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

OUT_STEM   = 'fig5_sim2real_font_changed'

Y_CLIP     = (-120, 40)
GOOD_COLOR = '#e8f4ea'
BAD_COLOR  = '#fbeceb'

BOX_W_D3 = 1.6
BOX_W_D5 = 2.8
PLATFORM_X_OFFSET = {'Forte-1': -2.6, 'Heron R3': 0.0, 'Nighthawk': 2.6}


def draw_box(ax, df):
    """df: one row per cluster, with column 'runs' = list of ECR_hw values."""
    for _, r in df.iterrows():
        vals      = np.asarray(r['runs'])
        is_d5     = r['Distance'] == 5
        w         = BOX_W_D5 if is_d5 else BOX_W_D3
        x0        = r['ECR_sim'] + PLATFORM_X_OFFSET[r['platform']]
        color     = COLORS[r['model']]
        vis       = np.clip(vals, Y_CLIP[0] + 3, Y_CLIP[1])
        clipped_any = (vals < Y_CLIP[0] + 3).any()

        if len(vals) == 1:
            ax.scatter(x0, vis[0],
                       marker=PLATFORM_MARKER[r['platform']],
                       color=color, s=120 if is_d5 else 70,
                       alpha=0.92, edgecolors='black', linewidth=0.9, zorder=4)
        else:
            q1, med, q3 = np.percentile(vis, [25, 50, 75])
            lo, hi      = vis.min(), vis.max()
            cap_w       = w * 0.4
            # Whiskers
            ax.plot([x0, x0], [lo, q1], color='black', lw=0.9, alpha=0.85, zorder=3)
            ax.plot([x0, x0], [q3, hi], color='black', lw=0.9, alpha=0.85, zorder=3)
            ax.plot([x0 - cap_w, x0 + cap_w], [lo, lo],
                    color='black', lw=0.9, alpha=0.85, zorder=3)
            ax.plot([x0 - cap_w, x0 + cap_w], [hi, hi],
                    color='black', lw=0.9, alpha=0.85, zorder=3)
            # Box
            ax.add_patch(Rectangle((x0 - w/2, q1), w, q3 - q1,
                                   facecolor=color, edgecolor='black',
                                   linewidth=0.8, alpha=0.85, zorder=4))
            # Median
            ax.plot([x0 - w/2, x0 + w/2], [med, med],
                    color='white', lw=1.8, zorder=5, solid_capstyle='butt')
            ax.plot([x0 - w/2, x0 + w/2], [med, med],
                    color='black', lw=0.5, alpha=0.8, zorder=6, solid_capstyle='butt')

        if clipped_any:
            ax.annotate('', xy=(x0, Y_CLIP[0] + 1.5),
                        xytext=(x0, Y_CLIP[0] + 7),
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        lw=1.2, alpha=0.8), zorder=5)


# ---------- Load & merge ----------
sim = load_sim()
hw  = load_hw_all(heron_pool=True, include_boston=True)
hw  = hw[~hw['is_hybrid']]                                      # standalone ML only

merged = hw.merge(sim, on=['code', 'Distance', 'noise', 'model'], how='inner')

# Cluster: per (model, distance, noise, platform, ECR_sim) collect run list
box_df = (merged.groupby(['model', 'Distance', 'noise', 'platform', 'ECR_sim'])
                ['ECR_hw'].apply(list).reset_index()
                .rename(columns={'ECR_hw': 'runs'}))

print(f"\n{'='*60}\nFigure 5 data summary\n{'='*60}")
print(f"Total clusters       : {len(box_df)}")
print(f"Total HW runs plotted: {sum(len(r) for r in box_df['runs'])}")
print(f"\nClusters per (platform, distance):")
print(box_df.groupby(['platform', 'Distance']).size().unstack(fill_value=0))
print(f"\nRuns per cluster (sanity):")
box_df['n_runs'] = box_df['runs'].apply(len)
print(box_df.groupby(['platform', 'Distance'])['n_runs'].agg(['min', 'max']))


# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)

for ax, nk in zip(axes, NOISE_ORDER):
    # Region shading
    ax.axhspan(0, Y_CLIP[1], facecolor=GOOD_COLOR, alpha=0.55, zorder=0)
    ax.axhspan(Y_CLIP[0], 0, facecolor=BAD_COLOR,  alpha=0.35, zorder=0)
    # Reference line
    ax.axhline(0, color='black', lw=1.0, alpha=0.65, zorder=1)

    draw_box(ax, box_df[box_df['noise'] == nk])

    n_clip = sum((np.asarray(r) < Y_CLIP[0] + 3).sum()
                 for r in box_df.loc[box_df['noise'] == nk, 'runs'])
    pdp = re.search(r"dp([\d.]+)", nk).group(1)
    title = rf"$p_{{\mathrm{{dp}}}} = {pdp}$"
    if n_clip > 0:
        title += f"   ({n_clip} runs below axis)"
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Simulation ECR (%)', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.2, zorder=1)
    ax.set_xlim(-3, 100)
    ax.set_ylim(Y_CLIP)

    if ax is axes[0]:
        ax.text(2, 20, 'improvement\nover NoCorr',
                fontsize=18, color='#2a7a36', alpha=0.9, style='italic',
                va='center')
        ax.text(2, -55, 'worse than\nNoCorr',
                fontsize=18, color='#9b2a24', alpha=0.9, style='italic')

axes[0].set_ylabel(
    r'Hardware ECR (%) $= 100 \cdot (1 - \mathrm{LER}_{\mathrm{ML}}/\mathrm{LER}_{\mathrm{NoCorr}})$',
    fontsize=20)


# ---------- Legend ----------
dec_handles = [Line2D([0], [0], marker='s', linestyle='', color='w',
                      markerfacecolor=COLORS[m], markeredgecolor='black',
                      markersize=10, label=m) for m in MODELS]
plat_handles = [Line2D([0], [0], marker=PLATFORM_MARKER[p], linestyle='',
                       color='lightgray', markeredgecolor='black',
                       markersize=10,
                       label=f'{p} ({"left" if p=="Forte-1" else "center" if p=="Heron R3" else "right"})')
                for p in PLATFORM_MARKER]
dist_handles = [Line2D([0], [0], marker='s', linestyle='', color='lightgray',
                       markeredgecolor='black', markersize=sz, label=f'd = {d}')
                for sz, d in [(7, 3), (12, 5)]]

fig.legend(handles=dec_handles, title='Decoder',
           loc='center left', bbox_to_anchor=(1.005, 0.75),
           fontsize=18, frameon=False, title_fontsize=20)
fig.legend(handles=plat_handles, title='Platform (x-position in cluster)',
           loc='center left', bbox_to_anchor=(1.005, 0.32),
           fontsize=18, frameon=False, title_fontsize=20)
fig.legend(handles=dist_handles, title='Distance (box width)',
           loc='center left', bbox_to_anchor=(1.005, 0.07),
           fontsize=18, frameon=False, title_fontsize=20)

fig.suptitle('Sim-to-real gap: simulation ECR vs hardware ECR',
             fontsize=22, x=0.62, y=1.02)
plt.tight_layout(rect=[0, 0, 0.98, 1.0])

for ext in ('png', 'pdf'):
    out = DATA_DIR / f'{OUT_STEM}.{ext}'
    plt.savefig(out, dpi=220, bbox_inches='tight')
    print(f"Saved: {out}")
plt.close()
