"""
IBM 시뮬레이터/QPU Job 제출 및 결과 수집

두 가지 모드:
  1. simulator: FakeBackend를 사용한 로컬 노이즈 시뮬레이션 (무료, 빠름)
  2. qpu: 실제 IBM Eagle QPU에 제출 (큐 대기 필요)

FakeBackend는 실제 하드웨어 캘리브레이션 데이터를 기반으로
노이즈를 시뮬레이션하므로, 실제 QPU와 유사한 결과를 냅니다.
"""

import os
import sys
import json
from qiskit import QuantumCircuit, transpile

current_dir = os.path.dirname(os.path.abspath(__file__))
ionq_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(ionq_dir)
sys.path.append(root_dir)

from paths import ProjectPaths

PATHS = ProjectPaths(root_dir)


class IBMSimulator:
    """
    IBM 백엔드에 회로를 제출하고 결과를 수집합니다.

    Parameters:
        backend_type: "simulator" (FakeBackend) 또는 "qpu" (실제 QPU)
        backend_name: IBM 백엔드 이름 (예: "ibm_yonsei")
    """

    def __init__(self, backend_type: str = "simulator", backend_name: str = "ibm_yonsei"):
        self.backend_type = backend_type
        self.backend_name = backend_name
        self.backend = None
        self.service = None

        self._initialize_backend()

    def _initialize_backend(self):
        keys = PATHS.load_keys()

        if self.backend_type == "simulator":
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()
            print(f"[IBMSimulator] Backend: AerSimulator (no noise, pipeline test)")

        elif self.backend_type == "qpu":
            from qiskit_ibm_runtime import QiskitRuntimeService
            api_key = keys.get("ibm_api_key", "")
            crn = keys.get("ibm_crn", "")

            if not api_key:
                raise ValueError("ibm_api_key not found in keys.json")

            self.service = QiskitRuntimeService(token=api_key, instance=crn)
            self.backend = self.service.backend(self.backend_name)
            print(f"[IBMSimulator] Backend: QPU ({self.backend_name}, {self.backend.num_qubits} qubits)")

        else:
            raise ValueError(f"Unknown backend_type: {self.backend_type}")

    def _get_fake_backend(self):
        """backend_name에 맞는 FakeBackend를 반환합니다."""
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeBrisbane

        fake_map = {
            "ibm_sherbrooke": FakeSherbrooke,
            "ibm_brisbane": FakeBrisbane,
        }

        if self.backend_name in fake_map:
            return fake_map[self.backend_name]()

        # 기본: 127큐빗 Eagle 계열
        print(f"    [Warning] No FakeBackend for '{self.backend_name}'. Using FakeSherbrooke.")
        return FakeSherbrooke()

    def run(self, circuit: QuantumCircuit, shots: int = 1000) -> dict:
        """
        회로를 IBM 백엔드에 제출하고 결과를 반환합니다.

        Args:
            circuit: 실행할 Qiskit 회로
            shots: 실행 횟수

        Returns:
            dict: {bitstring: count}
        """
        if self.backend_type == "simulator":
            # AerSimulator는 transpile 불필요
            job = self.backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts(circuit)

        elif self.backend_type == "qpu":
            # 실제 QPU: SamplerV2 사용
            from qiskit_ibm_runtime import SamplerV2 as Sampler

            # Transpile
            transpiled = transpile(circuit, backend=self.backend, optimization_level=1)

            print(f"[IBMSimulator] Submitting (shots={shots}, qubits={transpiled.num_qubits}, "
                f"depth={transpiled.depth()})")

            sampler = Sampler(backend=self.backend)
            job = sampler.run([transpiled], shots=shots)
            print(f"[IBMSimulator] Job ID: {job.job_id()}")
            print(f"[IBMSimulator] Waiting for results...")
            result = job.result()

            # SamplerV2 결과 → counts dict 변환
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()

        else:
            # FakeBackend / Aer: 직접 실행
            job = self.backend.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts(transpiled)

        print(f"[IBMSimulator] Completed. Unique outcomes: {len(counts)}")
        return counts

    def get_backend_info(self) -> dict:
        return {
            "backend_type": self.backend_type,
            "backend_name": self.backend_name,
            "num_qubits": getattr(self.backend, 'num_qubits', None),
        }


if __name__ == "__main__":
    print("=== IBM Simulator Test ===")

    # FakeBackend 테스트
    runner = IBMSimulator(backend_type="simulator", backend_name="ibm_sherbrooke")
    print(f"Backend info: {runner.get_backend_info()}")

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    counts = runner.run(qc, shots=100)
    print(f"Bell State: {counts}")
