"""
Simulation LER 재계산 (QuBench 후속 검증).

저장된 checkpoint로 held-out test shot에 inference만 수행하여 (재학습 없음)
각 (code, distance, noise profile, error rate, model)의 simulation LER를 계산한다.

정의:
  predicted mask  = (model logits > 0)          # run_stim_simulation.py와 동일 threshold
  residual        = predicted mask XOR true injected error mask (dataset labels)
  logical flip    = residual이 logical-Z support와 홀수번 겹치면 1
  sim LER         = mean(logical flip)

Baselines:
  NoCorrection = parity(labels) over logical support
  MWPM         = 하드웨어 파이프라인과 동일 계열의 cumulative-Z-syndrome MWPM
                 (heavyhex: MWPMHeavyHexDecoder, color: MWPMColorCodeDecoder,
                  surface: pymatching.Matching(H_z) with SurfaceCodeCircuit.z_stabilizers)

Logical-Z support (하드웨어 파이프라인 정의 재사용, 자체 유도 없음):
  surface : SurfaceCodeCircuit.logical_z (left column)
  heavyhex: HEAVYHEX_D3["logical_z"] = [0,3,6]
  color d5: D5_COLOR_CODE["logical_z"] = [0,1,2,3,4] (faces가 sim과 동일)
  color d3: sim(COLORCODE_FACES)과 hw(STEANE_CODE) 라벨링이 다르므로,
            face-구조 매칭으로 유일 permutation을 계산해 hw logical_z를
            sim 라벨링으로 pull-back (런타임 검증 포함). ※ 보고서에 flag됨.

사용법:
  python3 stim_simulation/compute_sim_ler.py            # 전체 실행
  python3 stim_simulation/compute_sim_ler.py --sanity   # 단일 config sanity check만
"""

import os
import sys
import json
import time
import argparse
import csv

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
IBM_DIR = os.path.join(ROOT_DIR, "ibm_experiment")
IONQ_DIR = os.path.join(ROOT_DIR, "ionq_experiment")
STIM_SIM_DIR = os.path.join(CURRENT_DIR, "simulation")
for p in (CURRENT_DIR, ROOT_DIR, IBM_DIR, IONQ_DIR, STIM_SIM_DIR):
    if p not in sys.path:
        sys.path.append(p)

from models.cnn import CNN
from models.gcn import GCN
from models.gcnii import GCNII
from models.gat import GAT
from models.appnp import APPNP
from models.gnn import GNN
from models.graph_transformer import GraphTransformer
from models.graph_mamba import GraphMamba
from paths import ProjectPaths

PATHS = ProjectPaths(ROOT_DIR)

# ==============================================================================
# CLI / 설정
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--sanity", action="store_true", help="단일 config sanity check만 수행")
parser.add_argument("-g", "--gpu", type=int, default=1)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--out", type=str,
                    default=os.path.join(CURRENT_DIR, "results", "gathered_sim_ler.csv"))
ARGS = parser.parse_args()

DEVICE = f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu"

with open(PATHS.stim_config()) as f:
    CONFIG = json.load(f)
MODEL_CONFIGS = CONFIG["models"]

NOISE_PROFILES = [
    "realistic/dp0.001_mf0.01_rf0.01_gd0.008",
    "realistic/dp0.005_mf0.02_rf0.02_gd0.015",
    "realistic/dp0.01_mf0.05_rf0.05_gd0.01",
]
ERROR_RATES = [0.005, 0.01, 0.05]
MODELS = ["CNN", "GCN", "GCNII", "GAT", "APPNP", "GNN", "GraphTransformer", "GraphMamba"]

# (code, distance) 조합: heavyhex는 d=3만 존재 (dataset/checkpoint 모두)
CODE_DISTANCES = [
    ("surface_code", 3), ("surface_code", 5),
    ("color_code", 3), ("color_code", 5),
    ("heavyhex_surface_code", 3),
]


# ==============================================================================
# Code별 구조 정보 (기존 코드에서 그대로 가져옴)
# ==============================================================================
def _load_ionq_modules():
    """
    ibm_experiment와 ionq_experiment 모두 `circuits`/`decoders` 패키지명을 쓰므로
    (ibm이 sys.path에서 우선), ionq 모듈은 파일 경로로 직접 로드하고
    decoder 내부의 `from circuits.qiskit_colorcode_generator import ...`가
    동작하도록 sys.modules에 해당 이름을 등록한다.
    """
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    if "circuits.qiskit_colorcode_generator" not in sys.modules:
        qcg = load("circuits.qiskit_colorcode_generator",
                   os.path.join(IONQ_DIR, "circuits", "qiskit_colorcode_generator.py"))
    else:
        qcg = sys.modules["circuits.qiskit_colorcode_generator"]
    if "ionq_mwpm_colorcode_decoder" not in sys.modules:
        dec = load("ionq_mwpm_colorcode_decoder",
                   os.path.join(IONQ_DIR, "decoders", "mwpm_colorcode_decoder.py"))
    else:
        dec = sys.modules["ionq_mwpm_colorcode_decoder"]
    return qcg, dec


def _colorcode_d3_permutation():
    """
    sim(COLORCODE_FACES[3])과 hw(STEANE_CODE) face 정의가 다르므로,
    face-membership 구조로 유일한 qubit relabeling pi(sim -> hw)를 계산한다.
    pi[q_sim] = q_hw, faces는 순서대로 대응 (sim face k <-> hw Z-stab row k).
    """
    from generators.color_code import COLORCODE_FACES
    STEANE_CODE = _load_ionq_modules()[0].STEANE_CODE
    sim_faces = COLORCODE_FACES[3]
    hw_faces = [s["qubits"] for s in STEANE_CODE["stabilizers"] if s["type"] == "Z"]
    n = 7

    def membership(faces, q):
        return frozenset(i for i, f in enumerate(faces) if q in f)

    hw_by_sig = {}
    for q in range(n):
        hw_by_sig.setdefault(membership(hw_faces, q), []).append(q)

    pi = [None] * n
    for q in range(n):
        sig = membership(sim_faces, q)
        cands = hw_by_sig.get(sig, [])
        if len(cands) != 1:
            raise RuntimeError(f"color d3 relabeling not unique for sim qubit {q}: {cands}")
        pi[q] = cands[0]
    assert sorted(pi) == list(range(n))

    # 검증: pi로 옮긴 sim face == hw face (순서대로)
    for k, f in enumerate(sim_faces):
        assert sorted(pi[q] for q in f) == sorted(hw_faces[k]), f"face {k} mismatch"
    return pi


def get_code_info(code_type, d):
    """
    반환 dict:
      rounds, num_stab(라운드당), z_slice(라운드 내 Z-stab 위치),
      logical(sim 라벨링의 logical-Z support), num_data,
      mwpm(cumulative Z syndrome (N, n_z) -> corrections (N, num_data))
    """
    import pymatching

    if code_type == "surface_code":
        from circuits.qiskit_surface_code_generator import SurfaceCodeCircuit
        sc = SurfaceCodeCircuit(distance=d, num_rounds=d)
        n_x, n_z = len(sc.x_stabilizers), len(sc.z_stabilizers)
        h_z = np.zeros((n_z, d * d), dtype=np.uint8)
        for i, stab in enumerate(sc.z_stabilizers):
            h_z[i, stab] = 1
        matcher = pymatching.Matching(h_z)

        def mwpm(z_cum):
            return matcher.decode_batch(z_cum.astype(np.uint8)).astype(np.uint8)

        return dict(rounds=d, num_stab=n_x + n_z, z_slice=slice(n_x, n_x + n_z),
                    logical=list(sc.logical_z), num_data=d * d, mwpm=mwpm)

    if code_type == "heavyhex_surface_code":
        assert d == 3
        from generators.heavyhex_surface_code import HEAVYHEX_D3
        from decoders.mwpm_heavyhex_decoder import MWPMHeavyHexDecoder
        dec = MWPMHeavyHexDecoder(distance=3)
        # 라운드당 측정 순서: [Z(4), X(4)] (generator 참조)
        return dict(rounds=3, num_stab=8, z_slice=slice(0, 4),
                    logical=list(HEAVYHEX_D3["logical_z"]), num_data=9,
                    mwpm=lambda z: dec.decode_z_syndrome(z.astype(np.uint8)).astype(np.uint8))

    if code_type == "color_code":
        from generators.color_code import COLORCODE_FACES, COLORCODE_NUM_DATA
        qcg, dec_mod = _load_ionq_modules()
        STEANE_CODE, D5_COLOR_CODE = qcg.STEANE_CODE, qcg.D5_COLOR_CODE
        MWPMColorCodeDecoder = dec_mod.MWPMColorCodeDecoder
        faces = COLORCODE_FACES[d]
        nf = len(faces)
        num_data = COLORCODE_NUM_DATA[d]
        dec = MWPMColorCodeDecoder(distance=d)

        if d == 3:
            pi = _colorcode_d3_permutation()  # sim -> hw
            logical = sorted(q for q in range(num_data) if pi[q] in STEANE_CODE["logical_z"])
            # sim face 순서 == hw Z-stab row 순서 (permutation 검증에서 확인) -> 신드롬은 그대로
            inv = np.argsort(np.array(pi))  # hw -> sim

            def mwpm(z_cum, _dec=dec, _pi=np.array(pi)):
                cache = {}
                out = np.zeros((len(z_cum), num_data), dtype=np.uint8)
                for i, syn in enumerate(z_cum.astype(np.uint8)):
                    key = syn.tobytes()
                    if key not in cache:
                        corr_hw = _dec._decode_single(syn)
                        cache[key] = corr_hw[_pi].astype(np.uint8)  # sim_corr[q]=hw_corr[pi[q]]
                    out[i] = cache[key]
                return out
        else:
            logical = list(D5_COLOR_CODE["logical_z"])

            def mwpm(z_cum, _dec=dec):
                cache = {}
                out = np.zeros((len(z_cum), num_data), dtype=np.uint8)
                for i, syn in enumerate(z_cum.astype(np.uint8)):
                    key = syn.tobytes()
                    if key not in cache:
                        cache[key] = _dec._decode_single(syn).astype(np.uint8)
                    out[i] = cache[key]
                return out

        # sim 라벨링에서 logical이 모든 face(X-stab)와 commute하는지 검증
        for k, f in enumerate(faces):
            assert len(set(logical) & set(f)) % 2 == 0, \
                f"color d={d}: logical {logical} anticommutes with face {k} {f}"

        # 라운드당 측정 순서: [X faces(nf), Z faces(nf)] (generator 참조)
        return dict(rounds=d, num_stab=2 * nf, z_slice=slice(nf, 2 * nf),
                    logical=logical, num_data=num_data, mwpm=mwpm)

    raise ValueError(code_type)


# ==============================================================================
# 공통 유틸
# ==============================================================================
def logical_flip_rate(residual, logical):
    """residual (N, num_data) uint8 -> logical support parity가 홀수인 비율"""
    return float(residual[:, logical].sum(axis=1).astype(np.int64).__mod__(2).mean())


def extract_z_syndromes(graph_features, info):
    """
    graph node feature[0] = ML-format detector (round0 raw + temporal diffs).

    반환:
      z_last: (N, n_z) 마지막 라운드 raw Z 측정 (= detector XOR-sum; 하드웨어
              MWPMHeavyHexDecoder._to_cumulative_z_stim / MWPMColorCodeDecoder.decode와 동일)
      z_mv:   (N, n_z) 라운드별 raw Z 측정 (detector cumsum mod 2)의 majority vote.
              주입 에러는 t=0에 고정되어 raw Z syndrome이 전 라운드 동일해야 하므로,
              round별 측정 노이즈를 걸러내는 필터.
    """
    det = graph_features[:, :, 0].astype(np.uint8)
    N = det.shape[0]
    det = det.reshape(N, info["rounds"], info["num_stab"])
    raw = np.cumsum(det, axis=1) % 2          # raw[r] = det[0] ^ ... ^ det[r]
    z_raw = raw[:, :, info["z_slice"]]        # (N, rounds, n_z)
    z_last = z_raw[:, -1, :]
    z_mv = (z_raw.sum(axis=1) * 2 > info["rounds"]).astype(np.uint8)
    return z_last, z_mv


def get_model_instance(model_name, input_shape, num_qubits):
    """run_stim_simulation.py::get_model_instance와 동일 (필요 모델만)."""
    params = MODEL_CONFIGS[model_name]["params"]
    if model_name == "CNN":
        return CNN(height=input_shape[1], width=input_shape[2], in_channels=input_shape[0],
                   num_classes=num_qubits)
    if model_name == "GCN":
        return GCN(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                   hidden_dim=params["hidden_dim"], num_layers=params["num_layers"])
    if model_name == "GCNII":
        return GCNII(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                     hidden_dim=params["hidden_dim"], num_layers=params["num_layers"],
                     alpha=params["alpha"], theta=params["theta"], dropout=params["dropout"])
    if model_name == "GAT":
        return GAT(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                   hidden_dim=params["hidden_dim"], heads=params["heads"],
                   num_layers=params["num_layers"], dropout=params["dropout"])
    if model_name == "APPNP":
        return APPNP(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                     hidden_dim=params["hidden_dim"], K=params["K"], alpha=params["alpha"])
    if model_name == "GNN":
        return GNN(num_nodes=input_shape[0], in_channels=input_shape[1], num_qubits=num_qubits,
                   hidden_dim=params["hidden_dim"], num_layers=params["num_layers"])
    if model_name == "GraphTransformer":
        return GraphTransformer(num_nodes=input_shape[0], in_channels=input_shape[1],
                                num_qubits=num_qubits, d_model=params["d_model"],
                                num_heads=params["num_heads"], num_layers=params["num_layers"],
                                dropout=params["dropout"])
    if model_name == "GraphMamba":
        return GraphMamba(num_nodes=input_shape[0], in_channels=input_shape[1],
                          num_qubits=num_qubits, d_model=params["d_model"],
                          num_layers=params["num_layers"], dropout=params["dropout"])
    raise ValueError(model_name)


@torch.no_grad()
def predict_masks(model_name, weight_path, X, num_qubits, edge_index, batch_size):
    """
    checkpoint 로드 후 inference. threshold는 기존 평가 코드와 동일: logits > 0.
    반환: (N, num_qubits) uint8 predicted mask, 소요시간(s)
    """
    cfg = MODEL_CONFIGS[model_name]
    is_graph = (cfg["type"] == "graph")
    use_adj = cfg["use_adj"]

    ck = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]

    model = get_model_instance(model_name, tuple(X.shape[1:]), num_qubits)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()

    adj = None
    ei = None
    if is_graph:
        ei = torch.LongTensor(edge_index).to(DEVICE)
        if use_adj:
            from torch_geometric.utils import to_dense_adj
            num_nodes = X.shape[1]
            adj = to_dense_adj(ei, max_num_nodes=num_nodes)[0]
            adj = adj + torch.eye(num_nodes, device=DEVICE)
            adj = (adj > 0).float()

    N = X.shape[0]
    preds = np.zeros((N, num_qubits), dtype=np.uint8)
    t0 = time.time()
    bs = batch_size
    i = 0
    while i < N:
        try:
            xb = torch.from_numpy(X[i:i + bs]).float().to(DEVICE)
            out = model(xb, adj) if use_adj else (model(xb, ei) if is_graph else model(xb))
            preds[i:i + bs] = (out > 0).to(torch.uint8).cpu().numpy()
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(32, bs // 2)
            print(f"      [OOM] batch -> {bs}")
    dt = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return preds, dt, bs


# ==============================================================================
# 메인 루프
# ==============================================================================
def run_config(code_type, d, noise, p, info, writer, fout, sanity=False):
    rows = []
    # --- 데이터 로드 (graph / image 독립 생성된 test set) ---
    data = {}
    for dtype in ("graph", "image"):
        path = PATHS.stim_data(code_type, noise, dtype, "test", d, p, "X")
        if not os.path.exists(path):
            print(f"    !! missing dataset: {path}")
            return []
        z = np.load(path)
        data[dtype] = (z["features"], z["labels"].astype(np.uint8))

    logical = info["logical"]
    Xg, yg = data["graph"]
    Xi, yi = data["image"]
    edge_index = np.load(PATHS.stim_edge(code_type, noise, d))

    # --- No Correction (graph/image 둘 다: 독립 test set 간 cross-check) ---
    nc_g = logical_flip_rate(yg, logical)
    nc_i = logical_flip_rate(yi, logical)
    # 해석적 기대값: 주입 에러는 IID Bern(p) -> P(odd parity in |support|)
    k = len(logical)
    analytic = 0.5 * (1.0 - (1.0 - 2.0 * p) ** k)
    print(f"    NC(graph)={nc_g:.4f} NC(image)={nc_i:.4f} analytic={analytic:.4f}")
    rows.append([code_type, d, noise.split("/")[-1], p, "NoCorrection", f"{nc_g:.6f}", len(yg)])

    # --- MWPM (graph test set 기준) ---
    z_last, z_mv = extract_z_syndromes(Xg, info)
    for tag, z_syn in (("MWPM", z_last), ("MWPM_mv", z_mv)):
        t0 = time.time()
        mwpm_corr = info["mwpm"](z_syn)
        mwpm_ler = logical_flip_rate(mwpm_corr ^ yg, logical)
        print(f"    {tag}={mwpm_ler:.4f} ({time.time()-t0:.1f}s)")
        rows.append([code_type, d, noise.split("/")[-1], p, tag, f"{mwpm_ler:.6f}", len(yg)])

    if sanity:
        # sanity: zero-correction 경로가 NC와 정확히 일치하는지
        zero = np.zeros_like(yg)
        assert abs(logical_flip_rate(zero ^ yg, logical) - nc_g) < 1e-12
        print("    [sanity] zero-correction pipeline == NC parity  OK")

    # --- ML 모델들 ---
    for m in MODELS:
        cfg = MODEL_CONFIGS[m]
        X, y = (Xi, yi) if cfg["type"] == "image" else (Xg, yg)
        wpath = PATHS.stim_weight(code_type, noise, m, d, p, "X")
        if not os.path.exists(wpath):
            print(f"    !! missing checkpoint: {wpath}")
            rows.append([code_type, d, noise.split("/")[-1], p, m, "NA", len(y)])
            continue
        preds, dt, bs_used = predict_masks(m, wpath, X, y.shape[1], edge_index, ARGS.batch)
        assert preds.shape == y.shape, f"{m}: pred {preds.shape} vs label {y.shape}"
        ler = logical_flip_rate(preds ^ y, logical)
        print(f"    {m:<17} LER={ler:.4f}  ({dt:.1f}s, batch={bs_used})")
        rows.append([code_type, d, noise.split("/")[-1], p, m, f"{ler:.6f}", len(y)])

    for r in rows:
        writer.writerow(r)
    fout.flush()
    return rows


def main():
    t_start = time.time()
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    print(f"=== Simulation LER 재계산 ===\nDevice: {DEVICE} | batch(start): {ARGS.batch}")
    print(f"Output: {ARGS.out}")

    combos = CODE_DISTANCES
    profiles = NOISE_PROFILES
    rates = ERROR_RATES
    if ARGS.sanity:
        combos = [("surface_code", 3)]
        profiles = NOISE_PROFILES[1:2]
        rates = [0.01]
        print(">>> SANITY MODE: surface_code d=3, dp0.005 profile, p=0.01만 수행")

    # resume: 이미 완료된 (code, d, noise, p)는 건너뜀 (11 rows = NC+MWPM×2+8모델)
    done = set()
    if os.path.exists(ARGS.out) and not ARGS.sanity:
        import collections
        cnt = collections.Counter()
        with open(ARGS.out) as f:
            for row in csv.DictReader(f):
                cnt[(row["code"], int(row["distance"]), row["noise_profile"],
                     float(row["error_rate"]))] += 1
        done = {k for k, v in cnt.items() if v >= 11}
        if done:
            print(f"resume: {len(done)}개 완료된 config 스킵")

    mode = "a" if done else "w"
    with open(ARGS.out, mode, newline="") as fout:
        writer = csv.writer(fout)
        if mode == "w":
            writer.writerow(["code", "distance", "noise_profile", "error_rate", "model",
                             "sim_LER", "n_shots"])
        for code_type, d in combos:
            info = get_code_info(code_type, d)
            print(f"\n### {code_type} d={d} | logical_z={info['logical']} "
                  f"rounds={info['rounds']} num_stab/round={info['num_stab']}")
            for noise in profiles:
                for p in rates:
                    if (code_type, d, noise.split("/")[-1], p) in done:
                        print(f"  -- {noise} p={p} [skip: 완료됨]")
                        continue
                    print(f"  -- {noise} p={p}")
                    run_config(code_type, d, noise, p, info, writer, fout, sanity=ARGS.sanity)

    print(f"\n=== 완료: {(time.time()-t_start)/60:.1f} min ===")


if __name__ == "__main__":
    main()
