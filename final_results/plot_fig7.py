import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# IEEEtran conference figure typography:
#   - Times body font (Nimbus Roman / Liberation Serif are metric-compatible
#     fallbacks when Times New Roman is not installed).
#   - STIX math so equations match the Times body (not DejaVu).
#   - fonttype 42 embeds TrueType outlines; IEEE PDF eXpress rejects the
#     Type-3 fonts matplotlib emits by default.
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'Nimbus Roman',
                          'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset' : 'stix',
    'font.size'        : 7,      # IEEE caption size baseline
    'axes.titlesize'   : 8,
    'axes.labelsize'   : 8,
    'xtick.labelsize'  : 7,
    'ytick.labelsize'  : 7,
    'legend.fontsize'  : 7,
    'pdf.fonttype'     : 42,
    'ps.fonttype'      : 42,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('DATA_DIR', _HERE)   # folder holding the CSVs (script dir)
OUT = os.environ.get('OUT_PDF', os.path.join(_HERE, 'fig7_transfer.pdf'))

ML8 = ['CNN','GNN','GAT','GCN','GCNII','APPNP','GraphTransformer','GraphMamba']

# (panel title, csv filename, distance, (sim_code, sim_distance))
CONFIGS = [
    ('Forte-1 d3', 'combined_forte-1.csv',       3, ('color_code', 3)),
    ('Forte-1 d5', 'combined_forte-1.csv',       5, ('color_code', 5)),
    ('ibm_boston d3',       'combined_ibm_boston.csv',     3, ('heavyhex_surface_code', 3)),
    ('ibm_aachen d3',       'combined_ibm_aachen.csv',     3, ('heavyhex_surface_code', 3)),
    ('ibm_pittsburgh d3',   'combined_ibm_pittsburgh.csv', 3, ('heavyhex_surface_code', 3)),
    ('ibm_miami d3',          'combined_ibm_miami.csv',      3, ('surface_code', 3)),
    ('ibm_miami d5',          'combined_ibm_miami.csv',      5, ('surface_code', 5)),
]

sim = pd.read_csv(os.path.join(DATA_DIR, 'gathered_stim.csv'))
sim = sim[sim['Distance'].isin([3, 5])]
ecr = (sim.groupby(['code','Distance','model'])['Best_ECR(%)']
          .mean().unstack('model')[ML8])
order = list(ecr.mean(axis=0).sort_values(ascending=False).index)  # sim strong -> weak
xlab  = [('GT' if m=='GraphTransformer' else 'GM' if m=='GraphMamba' else m) for m in order]

def mwpm_label(models):
    return 'MWPM' if 'MWPM' in models else 'MWPM_Restriction'

def mean_ler(df, d, name):
    s = df[(df['Distance']==d) & (df['Model']==name)]['Logical_Error_Rate']
    return s.mean() if len(s) else np.nan

fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.0), sharex=True)
axes = axes.ravel()
x = np.arange(len(order))
gmax = 0.0   # track the global maximum so every panel shares one y range

for k, (lab, fn, d, (c, dd)) in enumerate(CONFIGS):
    ax = axes[k]
    df = pd.read_csv(os.path.join(DATA_DIR, fn))
    mw = mwpm_label(set(df['Model']))
    st = [mean_ler(df, d, m)            for m in order]
    mf = [mean_ler(df, d, f'MWPM+{m}')  for m in order]
    lf = [mean_ler(df, d, f'{m}+MWPM')  for m in order]
    nc  = mean_ler(df, d, 'No_Correction')
    mwv = mean_ler(df, d, mw)

    ax.plot(x, st, 'o-',  color='0.55', ms=3, lw=1.0, label='standalone')
    ax.plot(x, mf, 's--', color='0.3',  ms=3, lw=1.0, label='MWPM+X')
    ax.plot(x, lf, '*-',  color='0.2', ms=5, lw=1.0, label='X+MWPM')
    ax.axhline(mwv, color='black', lw=1.0, ls=(0, (1, 1)))
    ax.axhline(nc,  color='0.6',  lw=1.0, ls=(0, (4, 2)))
    # MWPM/NoCorr baselines are identified in the shared legend (cell 8);
    # no per-panel inline label needed (it overlapped the curves).
    ax.set_title(lab, fontsize=7, pad=3)
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=7, rotation=90)
    # sharex hides inner-row tick labels by default; force them on every
    # panel so the top row also shows the decoder names.
    ax.tick_params(labelsize=8, length=0, labelbottom=True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    vals = [v for v in st+mf+lf if not np.isnan(v)]
    gmax = max(gmax, max(vals))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))   # 0.2-unit y ticks

# one shared y range -> the 0.2 tick spacing has the same height in every panel
ytop = np.ceil(gmax * 1.05 / 0.2) * 0.2
for ax in axes[:len(CONFIGS)]:
    ax.set_ylim(0, ytop)

# 8th cell: legend
axes[7].axis('off')
h, l = axes[0].get_legend_handles_labels()
axes[7].legend(
    h + [plt.Line2D([0],[0], color='black', ls=(0,(1,1))),
         plt.Line2D([0],[0], color='0.6',  ls=(0,(4,2)))],
    l + ['MWPM baseline', 'NoCorr baseline'],
    fontsize=7, loc='center', frameon=False)

fig.supylabel('Logical Error Rate', fontsize=8.5)
fig.supxlabel('Decoders ordered by decreasing simulation ECR', fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT, bbox_inches='tight')
plt.savefig(OUT.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
print('wrote', OUT)