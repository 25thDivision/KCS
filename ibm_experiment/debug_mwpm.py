"""MWPM 디버깅: noiseless 검증 + shot-level NC vs MWPM 비교."""
import os, sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(root_dir)

from circuits.heavyhex_surface_code_depth7 import HeavyHexSurfaceCode
from simulators.ibm_simulator import IBMSimulator
from extractors.syndrome_extractor import SyndromeExtractor
from decoders.mwpm_heavyhex_decoder import MWPMHeavyHexDecoder
from extractors.stim_compat import StimFormatConverter

DISTANCE = 3
NUM_ROUNDS = 3
INITIAL_STATE = 0
SHOTS = 2000

print("=" * 70)
print("  Step 1: Build circuit + run on noiseless AerSimulator")
print("=" * 70)
sc = HeavyHexSurfaceCode(distance=DISTANCE, num_rounds=NUM_ROUNDS)
qc = sc.build_circuit(initial_state=INITIAL_STATE)
runner = IBMSimulator(backend_type="simulator")
counts = runner.run(qc, shots=SHOTS)

syn_indices = sc.get_syndrome_indices()
extractor = SyndromeExtractor(syn_indices)
syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)
logical_z = np.array(syn_indices["logical_z"], dtype=np.int64)
print(f"  logical_z = {logical_z.tolist()}")

print("\n" + "=" * 70)
print("  Step 2: MWPM decode")
print("=" * 70)
mwpm = MWPMHeavyHexDecoder(distance=DISTANCE)
mwpm_corr = mwpm.decode(syndromes, data_states)
z_syn = mwpm._to_cumulative_z_stim(syndromes)

# Per-unique-outcome shot expansion via shot_counts
def logical_value(states):
    return (states[:, logical_z].sum(axis=1) % 2).astype(np.int64)

nc_logical = logical_value(data_states)
mwpm_logical = logical_value((data_states.astype(np.uint8) ^ mwpm_corr.astype(np.uint8)))

nc_correct = (nc_logical == INITIAL_STATE)
mwpm_correct = (mwpm_logical == INITIAL_STATE)

# Weighted by shot_counts
nc_ler = 1.0 - (nc_correct * shot_counts).sum() / shot_counts.sum()
mwpm_ler = 1.0 - (mwpm_correct * shot_counts).sum() / shot_counts.sum()
print(f"  Total shots: {shot_counts.sum()}")
print(f"  NC   LER: {nc_ler:.4f}")
print(f"  MWPM LER: {mwpm_ler:.4f}  (noiseless → 0% expected)")

print("\n" + "=" * 70)
print("  Step 3: Shot-level comparison")
print("=" * 70)
damaged = nc_correct & ~mwpm_correct       # NC ok, MWPM wrong
helped  = ~nc_correct & mwpm_correct       # NC wrong, MWPM ok
n_dmg = (damaged * shot_counts).sum()
n_help = (helped * shot_counts).sum()
print(f"  MWPM이 망친 shots  : {n_dmg} (unique outcomes: {damaged.sum()})")
print(f"  MWPM이 살린 shots : {n_help} (unique outcomes: {helped.sum()})")

print("\n  -- 망친 outcome 샘플 --")
for i in np.where(damaged)[0][:10]:
    corr_logZ = int(mwpm_corr[i][logical_z].sum() % 2)
    data_logZ = int(data_states[i][logical_z].sum() % 2)
    print(f"  count={shot_counts[i]:4d}  z_syn={z_syn[i].tolist()}  "
          f"data[logZ]={data_states[i][logical_z].tolist()}({data_logZ})  "
          f"corr={mwpm_corr[i].tolist()}  corr@logZ={corr_logZ}")

print("\n  -- 살린 outcome 샘플 --")
for i in np.where(helped)[0][:5]:
    print(f"  count={shot_counts[i]:4d}  z_syn={z_syn[i].tolist()}  "
          f"data={data_states[i].tolist()}  corr={mwpm_corr[i].tolist()}")

print("\n  -- z_syn=0 인데 data가 logical=1인 outcome (MWPM이 손 못 댐) --")
zero_syn_wrong = (z_syn.sum(axis=1) == 0) & ~nc_correct
for i in np.where(zero_syn_wrong)[0][:5]:
    print(f"  count={shot_counts[i]:4d}  data={data_states[i].tolist()}  "
          f"data[logZ]={data_states[i][logical_z].tolist()}")

print("\n" + "=" * 70)
print("  Step 4: Verify hw_to_stim_detectors vs _to_cumulative_z_stim")
print("=" * 70)
converter = StimFormatConverter(distance=DISTANCE, num_rounds=NUM_ROUNDS)
stim_detectors = converter.hw_to_stim_detectors(syndromes)
mwpm_z_syn = mwpm._to_cumulative_z_stim(syndromes)

print(f"  stim_detectors shape: {stim_detectors.shape}  (expected ({syndromes.shape[0]}, {NUM_ROUNDS*8}))")
print(f"  mwpm_z_syn     shape: {mwpm_z_syn.shape}      (expected ({syndromes.shape[0]}, 4))")

stim_cube = stim_detectors.reshape(-1, NUM_ROUNDS, 8).astype(np.uint8)
cum_from_stim = stim_cube.sum(axis=1) % 2
z_from_stim = cum_from_stim[:, :4]
x_from_stim = cum_from_stim[:, 4:]

match = np.array_equal(z_from_stim, mwpm_z_syn)
x_zero = np.all(x_from_stim == 0)
print(f"  Z-bit cumulative 일치: {match}")
print(f"  X-bit masked to 0    : {x_zero}  (reorder_hw_to_stim이 X를 0 마스킹)")

print("\n  -- 처음 5개 outcome 비교 --")
for i in range(min(5, syndromes.shape[0])):
    print(f"  [{i}] z_from_stim={z_from_stim[i].tolist()}  mwpm_z_syn={mwpm_z_syn[i].tolist()}  "
          f"{'OK' if np.array_equal(z_from_stim[i], mwpm_z_syn[i]) else 'MISMATCH'}")

assert match and x_zero, "검증 실패"
print("\n  ✓ 검증 통과: 두 함수의 Z-syndrome 누적 결과가 일치합니다.")