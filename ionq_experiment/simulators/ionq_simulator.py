"""
IonQ 시뮬레이터/QPU Job 제출 및 결과 수집
 
Qiskit-IonQ Provider를 사용하여:
  - IonQ 시뮬레이터 (noise_model="aria-1", "forte-1", 향후 "tempo")
  - IonQ QPU (Phase 3에서 사용)
에 회로를 제출하고 결과를 수집합니다.
"""
 
import os
import sys
import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
 
 
def _load_ionq_api_key() -> str:
    """keys.json에서 IonQ API 키를 로드합니다."""
    key_file = os.path.join(root_dir, "keys.json")
    if not os.path.exists(key_file):
        raise FileNotFoundError(
            f"keys.json not found at {key_file}. "
            f"Please create it with: {{\"ionq_api_key\": \"your_key_here\"}}"
        )
    with open(key_file, "r") as f:
        data = json.load(f)
    
    api_key = data.get("ionq_api_key", "")
    if not api_key:
        raise ValueError("ionq_api_key is empty in keys.json")
    return api_key
 
 
class IonQSimulator:
    """
    IonQ 시뮬레이터/QPU에 회로를 제출하고 결과를 수집합니다.
    """
    
    QUBIT_LIMITS = {
        "ideal": 29,
        "aria-1": 25,
        "aria-2": 25,
        "forte-1": 29,
        "forte-enterprise-1": 29,
    }
    
    def __init__(self, backend_type: str = "simulator", 
                 noise_model: str = "forte-1",
                 seed: int = None):
        self.backend_type = backend_type
        self.noise_model = noise_model
        self.seed = seed
        self.max_qubits = self.QUBIT_LIMITS.get(noise_model, 29)
        
        self.api_key = _load_ionq_api_key()
        self.provider = None
        self.backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        try:
            from qiskit_ionq import IonQProvider
        except ImportError:
            raise ImportError("qiskit-ionq package is required. Install with: pip install qiskit-ionq")
        
        self.provider = IonQProvider(token=self.api_key)
        
        if self.backend_type == "simulator":
            self.backend = self.provider.get_backend("ionq_simulator")
            if self.noise_model != "ideal":
                self.backend.set_options(noise_model=self.noise_model)
                if self.seed is not None:
                    self.backend.set_options(sampler_seed=self.seed)
            print(f"[IonQSimulator] Backend: simulator (noise_model={self.noise_model}, max_qubits={self.max_qubits})")
            
        elif self.backend_type == "qpu":
            qpu_map = {"aria-1": "ionq_qpu.aria-1", "forte-1": "ionq_qpu.forte-1"}
            qpu_name = qpu_map.get(self.noise_model)
            if not qpu_name:
                raise ValueError(f"No QPU mapping for noise_model={self.noise_model}")
            self.backend = self.provider.get_backend(qpu_name)
            print(f"[IonQSimulator] Backend: QPU ({qpu_name})")
        else:
            raise ValueError(f"Unknown backend_type: {self.backend_type}")
    
    def run(self, circuit: QuantumCircuit, shots: int = 1000) -> dict:
        has_mcm = any(
            instr.operation.name in ('reset', 'if_else')
            for instr in circuit.data
            if instr.operation.name not in ('barrier', 'measure')
        )

        if has_mcm:
            return self._run_aer_fallback(circuit, shots)

        # ── d=3: 기존 IonQ 클라우드 시뮬레이터 경로 ──
        transpiled = transpile(circuit, backend=self.backend, optimization_level=0)

        if transpiled.num_qubits > self.max_qubits:
            raise ValueError(
                f"Transpiled circuit uses {transpiled.num_qubits} qubits, "
                f"but {self.noise_model} supports max {self.max_qubits}."
            )

        print(f"[IonQSimulator] Submitting job (shots={shots}, qubits={transpiled.num_qubits})...")
        print(f"[IonQSimulator] Circuit: depth={transpiled.depth()}, gates={dict(transpiled.count_ops())}")

        job = self.backend.run(transpiled, shots=shots)
        print(f"[IonQSimulator] Job ID: {job.job_id()}")
        print(f"[IonQSimulator] Waiting for results...")

        result = job.result()
        counts = result.get_counts(transpiled)
        print(f"[IonQSimulator] Completed. Unique outcomes: {len(counts)}")

        return counts

    def _run_aer_fallback(self, circuit: QuantumCircuit, shots: int) -> dict:
        """
        d=5 ancilla reuse 회로용 로컬 Aer 시뮬레이션.
        IonQ API가 mid-circuit measurement를 지원하지 않으므로
        Forte-1 공개 스펙 기반 노이즈 모델로 로컬 실행.
        """
        noise_model = self._build_forte1_noise_model()
        aer_backend = AerSimulator(noise_model=noise_model)
        transpiled = transpile(circuit, backend=aer_backend, optimization_level=0)

        print(f"[IonQSimulator] MCM detected → AerSimulator fallback (forte-1 noise)")
        print(f"[IonQSimulator] Circuit: qubits={transpiled.num_qubits}, "
            f"depth={transpiled.depth()}, gates={dict(transpiled.count_ops())}")

        result = aer_backend.run(transpiled, shots=shots).result()
        counts = result.get_counts(transpiled)
        print(f"[IonQSimulator] Aer completed. Unique outcomes: {len(counts)}")

        return counts

    def _build_forte1_noise_model(self) -> NoiseModel:
        """IonQ Forte-1 공개 캘리브레이션 기반 노이즈 모델."""
        noise_model = NoiseModel()

        # Forte-1 specs (ionq.com/quantum-systems/forte-enterprise)
        err_1q = 0.0002   # 1Q RB error ~0.02%
        err_2q = 0.004    # 2Q DRB error ~0.4%
        err_spam = 0.005  # SPAM error  ~0.5%

        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(err_1q, 1), ['h', 'x', 'y', 'z', 'ry', 'rz', 's', 'sdg']
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(err_2q, 2), ['cx']
        )
        ro_err = ReadoutError(
            [[1 - err_spam, err_spam],
            [err_spam, 1 - err_spam]]
        )
        noise_model.add_all_qubit_readout_error(ro_err)

        return noise_model
    
    def get_backend_info(self) -> dict:
        return {
            "backend_type": self.backend_type,
            "noise_model": self.noise_model,
            "max_qubits": self.max_qubits,
            "seed": self.seed,
        }
 