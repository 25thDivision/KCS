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
        """
        회로를 IonQ에 제출하고 결과를 반환합니다.
        optimization_level=0으로 트랜스파일하여 게이트 재합성을 방지합니다.
        """
        # 트랜스파일 (optimization_level=0: 재합성 방지)
        transpiled = transpile(circuit, backend=self.backend, optimization_level=0)
        
        # 큐빗 수 검증
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
    
    def get_backend_info(self) -> dict:
        return {
            "backend_type": self.backend_type,
            "noise_model": self.noise_model,
            "max_qubits": self.max_qubits,
            "seed": self.seed,
        }
 