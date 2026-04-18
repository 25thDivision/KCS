"""
KCS 프로젝트 통합 경로 유틸리티

모든 경로는 프로젝트 루트(KCS/)로부터 동적으로 조합됩니다.

noise 파라미터에는 config의 noise_profiles key가 그대로 들어갑니다.
예: "realistic/dp0_mf0.005_rf0.005_gd0.004"

생성되는 경로 예시:
  dataset:  stim_simulation/dataset/color_code/realistic/dp0_mf0.005_rf0.005_gd0.004/graph/train_d3_p0.005_X.npz
  weight:   stim_simulation/saved_weights/color_code/realistic/dp0_mf0.005_rf0.005_gd0.004/GraphMamba/best_GraphMamba_d3_p0.005_X.pth
  result:   stim_simulation/test_results/color_code/realistic/dp0_mf0.005_rf0.005_gd0.004/benchmark_GraphMamba.csv
"""

import os
import json


class ProjectPaths:
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = root_dir

    # =========================================================================
    # 기본 디렉토리
    # =========================================================================
    @property
    def stim_dir(self) -> str:
        return os.path.join(self.root, "stim_simulation")

    @property
    def ionq_dir(self) -> str:
        return os.path.join(self.root, "ionq_experiment")

    @property
    def ibm_dir(self) -> str:
        return os.path.join(self.root, "ibm_experiment")

    @property
    def keys_path(self) -> str:
        return os.path.join(self.root, "keys.json")
    
    @property
    def logs_dir(self) -> str:
        return os.path.join(self.root, "logs")

    # =========================================================================
    # stim_simulation 경로
    # =========================================================================
    def stim_data_dir(self, code_type: str, noise: str, data_type: str) -> str:
        """stim_simulation/dataset/{code}/{noise}/graph|image/"""
        path = os.path.join(self.stim_dir, "dataset", code_type, noise, data_type)
        os.makedirs(path, exist_ok=True)
        return path

    def stim_data(self, code_type: str, noise: str, data_type: str,
                  prefix: str, d: int, p: float, err_type: str) -> str:
        return os.path.join(
            self.stim_data_dir(code_type, noise, data_type),
            f"{prefix}_d{d}_p{p}_{err_type}.npz"
        )

    def stim_edge(self, code_type: str, noise: str, d: int) -> str:
        return os.path.join(
            self.stim_data_dir(code_type, noise, "graph"),
            f"edges_d{d}.npy"
        )

    def stim_weight_dir(self, code_type: str, noise: str, model_name: str) -> str:
        """stim_simulation/saved_weights/{code}/{noise}/{model}/"""
        path = os.path.join(self.stim_dir, "saved_weights", code_type, noise, model_name)
        os.makedirs(path, exist_ok=True)
        return path

    def stim_weight(self, code_type: str, noise: str, model_name: str,
                    d: int, p: float, err_type: str) -> str:
        return os.path.join(
            self.stim_weight_dir(code_type, noise, model_name),
            f"best_{model_name}_d{d}_p{p}_{err_type}.pth"
        )

    def stim_result_dir(self, code_type: str, noise: str) -> str:
        """stim_simulation/results/{code}/{noise}/"""
        path = os.path.join(self.stim_dir, "results", code_type, noise)
        os.makedirs(path, exist_ok=True)
        return path

    def stim_result(self, code_type: str, noise: str, model_name: str) -> str:
        return os.path.join(
            self.stim_result_dir(code_type, noise),
            f"benchmark_{model_name}.csv"
        )

    def stim_config(self) -> str:
        return os.path.join(self.stim_dir, "config.json")

    # =========================================================================
    # experiment 경로 (ionq / ibm 공용)
    # =========================================================================
    def experiment_dir(self, platform: str) -> str:
        return os.path.join(self.root, f"{platform}_experiment")

    def experiment_config(self, platform: str) -> str:
        return os.path.join(self.experiment_dir(platform), "config.json")

    def experiment_result_dir(self, platform: str) -> str:
        path = os.path.join(self.experiment_dir(platform), "results")
        os.makedirs(path, exist_ok=True)
        return path

    # =========================================================================
    # 유틸리티
    # =========================================================================
    def load_keys(self) -> dict:
        if not os.path.exists(self.keys_path):
            return {}
        with open(self.keys_path, "r") as f:
            return json.load(f)

    def load_stim_config(self) -> dict:
        with open(self.stim_config(), "r") as f:
            return json.load(f)