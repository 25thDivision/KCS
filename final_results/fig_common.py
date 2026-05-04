"""
Common style / configuration shared by Fig 2 and Fig 5.
Import via:
    from fig_common import *
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl

# ---- IEEE conference template style (Times, 8pt base) ----
# Target physical sizes when printed in IEEEtran two-column:
#   1-column figure : 3.5 in (88.9 mm) wide
#   2-column figure : 7.16 in (181 mm) wide
# Per-figure fontsize= overrides may scale these up to compensate when the
# source figsize is larger than the printed size (matplotlib scales linearly).
mpl.rcParams.update({
    'font.family':           'serif',
    'font.serif':            ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset':      'stix',
    'font.size':             8,
    'axes.labelsize':        8,
    'axes.titlesize':        8,
    'xtick.labelsize':       7,
    'ytick.labelsize':       7,
    'legend.fontsize':       7,
    'legend.title_fontsize': 8,
    'figure.titlesize':      9,
    'axes.linewidth':        0.6,
    'pdf.fonttype':          42,   # IEEE Xplore requires TrueType embed
    'ps.fonttype':           42,
})

# ---- File paths ----
DATA_DIR = Path('.')
F_FORTE1     = 'combined_forte-1.csv'
F_MIAMI      = 'combined_ibm_miami.csv'
F_BOSTON     = 'combined_ibm_boston.csv'
F_AACHEN     = 'combined_ibm_aachen.csv'
F_PITTSBURGH = 'combined_ibm_pittsburgh.csv'
F_STIM       = 'gathered_stim.csv'

# ---- Decoder list ----
MODELS = ['APPNP', 'CNN', 'GAT', 'GCN', 'GCNII', 'GNN',
          'GraphMamba', 'GraphTransformer']

# ---- Colors: Attention family hot / others muted ----
COLORS = {
    'GraphTransformer': '#d62728',   # red
    'GraphMamba':       '#ff7f0e',   # orange
    'GAT':              '#bcbd22',   # olive
    'CNN':              '#2ca02c',   # green
    'GCN':              '#1f77b4',   # blue
    'GCNII':            '#17becf',   # cyan
    'APPNP':            '#9467bd',   # purple
    'GNN':              '#8c564b',   # brown
}

# ---- Platform markers ----
PLATFORM_MARKER = {'Forte-1': '^', 'Heron R3': 'o', 'Nighthawk': 's'}
PLATFORM_CODE   = {'Forte-1': 'color_code',
                   'Heron R3': 'heavyhex_surface_code',
                   'Nighthawk': 'surface_code'}

# ---- Noise profile labels ----
NOISE_LABEL = {
    'dp0.001_mf0.01_rf0.01_gd0.008': 'realistic',
    'dp0.005_mf0.02_rf0.02_gd0.015': 'heavy',
    'dp0.01_mf0.05_rf0.05_gd0.01':   'extreme',
}
NOISE_ORDER = list(NOISE_LABEL.keys())

# ---- Canonical training p ----
# Each HW row records `Stim_Error_Rate` = the depolarizing p used to train the
# decoder. Sim CSV's `Error_Rate(p)` is the same quantity. We pick p=0.005 as
# canonical: lowest p in the available set, closest to actual hardware regime.
# Higher training p (0.01, 0.05) produces pathological weights for several
# graph-conv architectures and distorts results if averaged across p.
CANONICAL_P = 0.005


# ============================================================
# Loaders
# ============================================================
def load_sim() -> pd.DataFrame:
    """Return one row per (code, distance, noise, model) at CANONICAL_P with best ECR."""
    s = pd.read_csv(DATA_DIR / F_STIM)
    s = s[s['Distance'] <= 5]                              # d=7 out of paper scope
    s = s[s['Error_Rate(p)'] == CANONICAL_P]
    sb = (s.groupby(['code', 'Distance', 'noise', 'model'])
            ['Best_ECR(%)'].max().reset_index()
            .rename(columns={'Best_ECR(%)': 'ECR_sim'}))
    return sb


def _hw_per_run(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """Return per-run rows for ML decoders only, with ECR and LER columns.
    NoCorr baseline is the average across NoCorr runs at the same distance."""
    base = (df[df['Model'] == 'No_Correction']
            .groupby('Distance')['Logical_Error_Rate'].mean())

    ml = df[df['Weight_Noise'].notna()
            & (df['Stim_Error_Rate'] == CANONICAL_P)
            & ~df['Model'].isin(['No_Correction', 'MWPM', 'MWPM_Restriction'])
            ].copy()
    ml['LER_nocorr'] = ml['Distance'].map(base)
    ml['ECR_hw']     = 100 * (1 - ml['Logical_Error_Rate'] / ml['LER_nocorr'])
    ml['platform']   = platform
    ml['noise']      = ml['Weight_Noise'].str.replace('realistic/', '', regex=False)
    ml['code']       = PLATFORM_CODE[platform]
    return ml.rename(columns={'Model': 'model_full',
                              'Logical_Error_Rate': 'LER_hw'})


def load_hw_all(heron_pool: bool = True, include_boston: bool = True) -> pd.DataFrame:
    """Concatenate all hardware CSVs into a long-format frame.
    `model_full` may include 'MWPM+...' prefixes; downstream consumers split."""
    parts = [_hw_per_run(pd.read_csv(DATA_DIR / F_FORTE1), 'Forte-1')]
    if heron_pool:
        heron_files = [F_AACHEN, F_PITTSBURGH] + ([F_BOSTON] if include_boston else [])
        heron_df = pd.concat([pd.read_csv(DATA_DIR / f) for f in heron_files],
                             ignore_index=True)
        parts.append(_hw_per_run(heron_df, 'Heron R3'))
    parts.append(_hw_per_run(pd.read_csv(DATA_DIR / F_MIAMI), 'Nighthawk'))
    out = pd.concat(parts, ignore_index=True)
    # Split decoder name
    out['is_hybrid'] = out['model_full'].str.startswith('MWPM+')
    out['model']     = out['model_full'].str.replace('MWPM+', '', regex=False)
    return out


def load_baselines(platform_files: dict) -> pd.DataFrame:
    """Per-run NoCorr / MWPM / MWPM_Restriction LER values (no Weight_Noise)."""
    rows = []
    for platform, fp in platform_files.items():
        df = pd.read_csv(DATA_DIR / fp)
        b = df[df['Model'].isin(['No_Correction', 'MWPM', 'MWPM_Restriction'])
               & df['Weight_Noise'].isna()].copy()
        b['platform'] = platform
        rows.append(b[['Model', 'Distance', 'Run', 'Logical_Error_Rate', 'platform']])
    return pd.concat(rows, ignore_index=True).rename(
        columns={'Model': 'model', 'Logical_Error_Rate': 'LER'})
