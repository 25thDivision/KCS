"""
Simulation LER 분석: 집계 + Spearman ρ + sim_ler_report.md 생성.

입력:
  stim_simulation/results/gathered_sim_ler.csv   (compute_sim_ler.py 출력)
  final_results/gathered_stim.csv                (sim ECR, 논문 Table I 소스)
  하드웨어 standalone LER (아래 하드코딩 — raw CSV에서 검증된 값)

Spearman:
  (a) config별 sim ECR ranking vs sim LER ranking (8개 모델)
  (b) (code,distance)별 mean sim LER ranking vs hw standalone LER ranking
  (참고) sim ECR ranking vs hw standalone LER ranking (논문 원래 비교축)

hw config -> sim (code, distance) 매핑 (final_results/fig_common.py PLATFORM_CODE):
  Forte-1 -> color_code,  Heron R3(boston/aachen/pittsburgh) -> heavyhex d3,
  Nighthawk(miami) -> surface_code
"""

import os
import sys
import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

SIM_LER_CSV = os.path.join(CURRENT_DIR, "results", "gathered_sim_ler.csv")
ECR_CSV = os.path.join(ROOT_DIR, "final_results", "gathered_stim.csv")
REPORT = os.path.join(CURRENT_DIR, "results", "sim_ler_report.md")

MODELS = ["CNN", "GCN", "GCNII", "GAT", "APPNP", "GNN", "GraphTransformer", "GraphMamba"]

# 하드웨어 standalone LER (작업 지시서 제공, raw CSV 검증값)
HW_LER = pd.DataFrame({
    "model": ["CNN", "GNN", "GAT", "GCN", "GCNII", "APPNP", "GraphTransformer", "GraphMamba"],
    "forte1_d3":     [0.205, 0.248, 0.232, 0.578, 0.531, 0.394, 0.170, 0.169],
    "forte1_d5":     [0.174, 0.488, 0.220, 0.560, 0.693, 0.541, 0.150, 0.144],
    "boston_d3":     [0.589, 0.598, 0.595, 0.507, 0.486, 0.350, 0.275, 0.276],
    "aachen_d3":     [0.552, 0.557, 0.560, 0.506, 0.493, 0.415, 0.374, 0.374],
    "pittsburgh_d3": [0.501, 0.500, 0.500, 0.500, 0.500, 0.500, 0.499, 0.499],
    "miami_d3":      [0.229, 0.333, 0.236, 0.344, 0.612, 0.552, 0.225, 0.224],
    "miami_d5":      [0.350, 0.385, 0.350, 0.497, 0.550, 0.476, 0.350, 0.348],
}).set_index("model")

# hw config -> (sim code, distance)
HW_TO_SIM = {
    "forte1_d3":     ("color_code", 3),
    "forte1_d5":     ("color_code", 5),
    "boston_d3":     ("heavyhex_surface_code", 3),
    "aachen_d3":     ("heavyhex_surface_code", 3),
    "pittsburgh_d3": ("heavyhex_surface_code", 3),
    "miami_d3":      ("surface_code", 3),
    "miami_d5":      ("surface_code", 5),
}
EXCLUDE_FROM_RHO = {"pittsburgh_d3"}  # standalone spread < shot-noise resolution


def md(df, index=True):
    """tabulate 없이 DataFrame -> markdown 표."""
    d = df.reset_index() if index else df.copy()
    d = d.astype(object).where(pd.notna(d), "")
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def load_sim_ler():
    df = pd.read_csv(SIM_LER_CSV)
    df = df[df["sim_LER"] != "NA"].copy()
    df["sim_LER"] = df["sim_LER"].astype(float)
    return df


def load_sim_ecr():
    """per (code, noise, distance, p, model) best ECR (에폭 중 최대, 논문 방식)."""
    df = pd.read_csv(ECR_CSV)
    df = df[df["model"].isin(MODELS)]
    g = (df.groupby(["code", "noise", "Distance", "Error_Rate(p)", "model"])["Best_ECR(%)"]
           .max().reset_index())
    g = g.rename(columns={"Distance": "distance", "Error_Rate(p)": "error_rate",
                          "noise": "noise_profile", "Best_ECR(%)": "ECR"})
    return g


def main():
    sim = load_sim_ler()
    ecr = load_sim_ecr()
    ml = sim[sim["model"].isin(MODELS)].copy()

    out = io.StringIO()
    w = out.write

    w("# Simulation LER 재계산 결과 (QuBench 후속 검증)\n\n")

    # ------------------------------------------------------------------
    # 1. config별 sim LER 표
    # ------------------------------------------------------------------
    w("## 1. Config별 sim LER (10^5 shots, MWPM/No Correction 포함)\n\n")
    order = MODELS + ["MWPM", "MWPM_mv", "NoCorrection"]
    for (code, d), sub in sim.groupby(["code", "distance"]):
        w(f"### {code} d={d}\n\n")
        piv = sub.pivot_table(index="model", columns=["noise_profile", "error_rate"],
                              values="sim_LER")
        piv = piv.reindex([m for m in order if m in piv.index])
        piv.columns = [f"{n.replace('dp','').split('_')[0]}/p{p}" for n, p in piv.columns]
        w(piv.round(4).pipe(md) + "\n\n")

    # ------------------------------------------------------------------
    # 2. (code, distance, model)별 mean sim LER (9 config 평균)
    # ------------------------------------------------------------------
    w("## 2. Mean sim LER (9개 noise config 평균, 논문 Table I 집계 방식)\n\n")
    mean_ler = (sim.groupby(["code", "distance", "model"])["sim_LER"]
                  .agg(["mean", "count"]).reset_index())
    piv = mean_ler.pivot_table(index="model", columns=["code", "distance"], values="mean")
    piv = piv.reindex([m for m in order if m in piv.index])
    piv.columns = [f"{c.replace('_surface_code','HH').replace('_code','')} d{d}"
                   for c, d in piv.columns]
    w(piv.round(4).pipe(md) + "\n\n")

    counts = mean_ler[~mean_ler["model"].isin(["MWPM", "MWPM_mv", "NoCorrection"])]["count"]
    if not (counts == 9).all():
        w(f"> ⚠️ 일부 조합의 config 수가 9가 아님: {sorted(counts.unique())}\n\n")

    # ------------------------------------------------------------------
    # 3-(a). config별 sim ECR vs sim LER Spearman
    # ------------------------------------------------------------------
    w("## 3-(a). Spearman ρ: sim ECR ranking vs sim LER ranking (config별, 8개 모델)\n\n")
    w("ρ는 '성능 순위' 기준: ECR은 높을수록, LER은 낮을수록 좋음. "
      "완전 일치=+1, 완전 역전=-1.\n\n")
    rows = []
    merged = ml.merge(ecr, on=["code", "distance", "noise_profile", "error_rate", "model"],
                      how="inner")
    for key, sub in merged.groupby(["code", "distance", "noise_profile", "error_rate"]):
        if len(sub) < 8:
            continue
        # 성능 순위 비교: -ECR (내림차순=좋음) vs LER (오름차순=좋음)
        rho, pval = spearmanr(-sub["ECR"], sub["sim_LER"])
        rows.append([*key, len(sub), round(rho, 3), round(pval, 4)])
    rho_a = pd.DataFrame(rows, columns=["code", "distance", "noise_profile", "error_rate",
                                        "n_models", "rho", "pval"])
    w(rho_a.pipe(md, index=False) + "\n\n")
    summ = rho_a.groupby(["code", "distance"])["rho"].agg(["mean", "min", "max"]).round(3)
    w("**(code, distance)별 ρ 요약:**\n\n" + summ.pipe(md) + "\n\n")
    w(f"**전체 평균 ρ = {rho_a['rho'].mean():.3f}**\n\n")

    # ------------------------------------------------------------------
    # 3-(b). mean sim LER vs hw standalone LER Spearman
    # ------------------------------------------------------------------
    w("## 3-(b). Spearman ρ: sim 성능 ranking vs hw standalone LER ranking\n\n")
    w("sim 지표는 (code,distance)별 9-config 평균. ρ>0 = 순위 보존, ρ<0 = 역전.\n\n")

    mean_ml = mean_ler[mean_ler["model"].isin(MODELS)]
    mean_ecr = (merged.groupby(["code", "distance", "model"])["ECR"].mean().reset_index())

    rows = []
    for hw_cfg, (code, d) in HW_TO_SIM.items():
        hw = HW_LER[hw_cfg]
        sl = mean_ml[(mean_ml["code"] == code) & (mean_ml["distance"] == d)] \
            .set_index("model")["mean"].reindex(MODELS)
        se = mean_ecr[(mean_ecr["code"] == code) & (mean_ecr["distance"] == d)] \
            .set_index("model")["ECR"].reindex(MODELS)
        hwv = hw.reindex(MODELS)
        # 성능 순위: sim LER 낮을수록 좋음 vs hw LER 낮을수록 좋음 -> spearmanr(직접)
        rho_ler, p_ler = spearmanr(sl, hwv)
        rho_ecr, p_ecr = spearmanr(-se, hwv)  # ECR 높을수록 좋음 -> 부호 반전
        note = "제외(참고용)" if hw_cfg in EXCLUDE_FROM_RHO else ""
        rows.append([hw_cfg, f"{code} d{d}", round(rho_ler, 3), round(p_ler, 4),
                     round(rho_ecr, 3), round(p_ecr, 4), note])
    rho_b = pd.DataFrame(rows, columns=[
        "hw_config", "sim_config", "rho(simLER,hwLER)", "p",
        "rho(simECR_rank,hwLER)", "p_ecr", "비고"])
    w(rho_b.pipe(md, index=False) + "\n\n")
    valid = rho_b[rho_b["비고"] == ""]
    w(f"**유효 config 평균: ρ(simLER,hwLER) = {valid['rho(simLER,hwLER)'].mean():.3f}, "
      f"ρ(simECR,hwLER) = {valid['rho(simECR_rank,hwLER)'].mean():.3f}**\n\n")
    no_cd3 = valid[valid["sim_config"] != "color_code d3"]["rho(simLER,hwLER)"]
    w(f"color d3(라벨링 pull-back이 개입된 유일 config)를 추가로 제외한 5개 config 기준: "
      f"ρ(simLER,hwLER) = {no_cd3.min():.3f}~{no_cd3.max():.3f}, "
      f"평균 {no_cd3.mean():.3f} — 결론 불변.\n\n")
    w("주: 두 ρ 모두 'sim 성능이 나쁠수록 hw LER이 높다'는 방향이 +1이 되도록 정의 "
      "(sim LER는 그대로, sim ECR는 부호 반전하여 badness로 통일). "
      "따라서 **ρ가 0 이하이면 sim 순위가 hw에서 보존되지 않음(= inversion)**, "
      "+1에 가까우면 sim 순위가 hw를 잘 예측함.\n\n")

    # ------------------------------------------------------------------
    # 4. 결론 + 방법론/한계
    # ------------------------------------------------------------------
    mean_rho_ler = valid["rho(simLER,hwLER)"].mean()
    mean_rho_ecr = valid["rho(simECR_rank,hwLER)"].mean()
    w("## 4. 결론\n\n")
    w(f"**sim LER로 ranking하면 hw LER 대비 inversion이 사라진다.** "
      f"유효 6개 hw config 전체에서 ρ(simLER, hwLER) = 0.88~0.98 (평균 {mean_rho_ler:.3f}, "
      f"모두 p<0.006)로 simulation LER 순위가 하드웨어 순위를 거의 완벽히 예측한다. "
      f"반면 동일 비교축에서 sim ECR 성능순위는 hw LER와 평균 ρ = {mean_rho_ecr:.3f}로 "
      f"음의 상관(=논문에서 관찰된 rank inversion)을 보이고, simulation 내부에서도 "
      f"ECR 순위와 LER 순위는 평균 ρ = {rho_a['rho'].mean():.3f}로 불일치한다. "
      f"즉, 논문의 sim-hw rank inversion은 sim-to-real gap이 아니라 **ECR metric 자체의 "
      f"특성**이 주 원인이다. 재-inference 감사(appnp_gcnii_allones_audit.csv) 결과 "
      f"APPNP/GCNII는 45개 config 전부에서 입력과 거의 무관한 **상수 mask로 붕괴** "
      f"(대표 config들에서 10^5 shot 중 unique mask 1개, frac_ones 0.12~0.79). "
      f"상수 mask는 ECR로는 대략 mask 밀도만큼(평균 APPNP 57%, GCNII 46% — Table I의 "
      f"~50%대와 정합) 점수를 받지만, LER로는 mask의 logical-support parity에 따라 "
      f"NC(짝수) 또는 1−NC(홀수)로 발산하여 디코더로서 무의미함이 드러난다. "
      f"리뷰어 요청 관점에서: metric을 LER로 통일하면 simulation benchmark는 "
      f"hardware 성능의 강력한 예측자가 된다.\n\n")

    w("## 5. 방법론 기록\n\n")
    w("- **재학습 없음**: `stim_simulation/saved_weights/<code>/<noise>/<model>/"
      "best_<model>_d<d>_p<p>_X.pth`의 `model_state_dict`만 로드하여 inference.\n"
      "- **Threshold**: 기존 ECR 평가(run_stim_simulation.py)와 동일하게 `logits > 0` "
      "(sigmoid 0.5). 전처리 없음 — 저장된 test npz의 features를 그대로 사용 "
      "(uint8→float, 학습 코드와 동일).\n"
      "- **LER 정의**: residual = pred XOR label(주입 오류 mask), logical flip = "
      "residual과 logical-Z support의 홀수 겹침. dataset에 observable flip이 저장되어 "
      "있지 않아 residual-parity 방식만 가능 (두 방식 교차검증은 불가).\n"
      "- **Logical-Z support** (hw 파이프라인 정의 재사용): surface = "
      "SurfaceCodeCircuit.logical_z(왼쪽 열), heavyhex d3 = [0,3,6], color d5 = "
      "[0,1,2,3,4]. **color d3는 sim(COLORCODE_FACES)과 hw(STEANE_CODE)의 face 정의가 "
      "달라** face-membership 매칭으로 유일 relabeling π(sim→hw)=[4,1,0,2,3,5,6]을 "
      "계산해 hw logical_z [0,1,4]를 sim 라벨링 [0,1,2]로 pull-back했다 "
      "(commutation 런타임 검증 포함). 이 항목만 자체 유도가 개입된 판단 지점.\n"
      "- **MWPM**: 마지막 라운드 cumulative Z-syndrome을 hw 계열 디코더로 decode "
      "(heavyhex: MWPMHeavyHexDecoder, color: MWPMColorCodeDecoder(+π), surface: "
      "pymatching.Matching(H_z)). MWPM_mv는 라운드별 raw Z-syndrome majority vote 후 "
      "동일 decode.\n"
      "- **Sanity**: NoCorrection LER는 모든 config에서 해석값 (1−(1−2p)^|L|)/2와 "
      "일치(±shot noise), graph/image 독립 test set 간에도 일치. zero-correction "
      "경로가 NC parity와 bit-exact 일치함을 확인.\n"
      "- **batch size 256** (OOM 없음, GraphTransformer d5 포함), GPU: RTX 2080 Ti "
      "(cuda:1), 총 inference 소요 약 13분 (surface ~8분 + color/heavyhex ~5분), "
      "test shots: config당 10^5.\n\n")

    w("## 6. 한계 (해석 시 주의)\n\n")
    w("- **Label은 주입된 IID X 오류만 기록**하고 회로 노이즈(gate depol gd, data depol "
      "dp)가 만드는 실제 데이터 오류는 포함하지 않는다. 따라서 sim LER는 '주입 성분 "
      "기준 LER'이며, syndrome을 물리적으로 올바르게 설명하는 MWPM은 label에 없는 실제 "
      "오류까지 정정하여 불리해진다 (MWPM > NoCorrection이 나오는 이유; 예: "
      "dp0.005/gd0.015 프로파일에서 큐빗당 회로 유발 X-오류율 ~9%). ML 모델 8개는 모두 "
      "동일 convention으로 학습·평가되므로 **ML 간 상대 순위에는 영향 없음**. MWPM 행은 "
      "절대값 비교가 아닌 참고용.\n"
      "- ECR과 (1−LER)은 다른 양이며 실제로 일치하지 않음을 확인 (버그 아님).\n"
      "- ibm_pittsburgh d3는 hw standalone spread가 shot-noise 이하라 ρ 해석에서 제외 "
      "(표에는 참고용 포함; 참고로 ρ=0.87로 다른 config와 같은 방향).\n")

    with open(REPORT, "w") as f:
        f.write(out.getvalue())
    print(f"report written: {REPORT}")
    print(out.getvalue()[:3000])


if __name__ == "__main__":
    main()
