"""
IBM 시뮬레이터/QPU Job 제출 및 결과 수집

두 가지 모드:
  1. simulator: 실제 IBM 백엔드(self.backend_name)의 calibration 스냅샷을 fetch
                해서 NoiseModel.from_backend로 노이즈를 입힌 AerSimulator로 로컬 실행.
                백엔드 fetch는 properties만 받으므로 크레딧 무소모. capture(신드롬
                박제)는 simulator 모드에서 항상 비활성.
  2. qpu:      실제 IBM QPU에 SamplerV2로 제출 (큐 대기 + 크레딧 소모).

두 분기 모두 self.hw_backend = 실제 IBM 백엔드 객체를 보유하며, transpile은
self.hw_backend의 coupling map / target / durations 기준으로 수행.
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

    def __init__(self, backend_type: str = "simulator", backend_instance: str = "Yonsei_internal", backend_name: str = "ibm_yonsei", capture: bool = False):
        self.backend_type = backend_type
        self.backend_instance = backend_instance
        self.backend_name = backend_name
        self.backend = None        # 회로를 실제 실행할 객체 (simulator → AerSimulator, qpu → IBM backend)
        self.hw_backend = None     # transpile/coupling/target 참조용 IBM 백엔드 (두 분기 모두 보유)
        self.service = None
        self.capture = capture

        self._initialize_backend()

    def _initialize_backend(self):
        keys = PATHS.load_keys()

        if self.backend_type == "simulator":
            # 실제 IBM 백엔드의 calibration을 fetch (properties only → 크레딧 무소모)
            # 후 NoiseModel.from_backend로 노이즈 입힌 AerSimulator를 self.backend로 사용.
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel

            api_key = keys.get("ibm_api_key", "")
            if not api_key:
                raise ValueError("ibm_api_key not found in keys.json")

            self.service = QiskitRuntimeService(token=api_key, instance=self.backend_instance)
            self.hw_backend = self.service.backend(self.backend_name)

            noise_model = NoiseModel.from_backend(self.hw_backend)
            self.backend = AerSimulator(
                noise_model=noise_model,
                coupling_map=self.hw_backend.coupling_map,
                basis_gates=noise_model.basis_gates,
            )
            self._log_simulator_setup()

        elif self.backend_type == "qpu" or self.backend_type == "QPU":
            from qiskit_ibm_runtime import QiskitRuntimeService
            api_key = keys.get("ibm_api_key", "")
            crn = keys.get("ibm_crn", "")

            if not api_key:
                raise ValueError("ibm_api_key not found in keys.json")

            self.service = QiskitRuntimeService(token=api_key, instance=self.backend_instance)
            self.backend = self.service.backend(self.backend_name)
            self.hw_backend = self.backend  # transpile은 hw_backend를 보며, qpu에선 둘이 같음
            print(f"[IBMSimulator] Backend: QPU ({self.backend_name}, {self.backend.num_qubits} qubits)")

        else:
            raise ValueError(f"Unknown backend_type: {self.backend_type}")

    def _log_simulator_setup(self):
        """simulator 모드 진입 시 calibration 시점 + 평균 1Q/2Q/readout error 한 줄 출력.

        avg error 산출 실패는 'n/a'로 폴백 (그 외 동작 무영향).
        """
        last_update = "unknown"
        avg_1q = avg_2q = avg_ro = None
        try:
            props = self.hw_backend.properties()
            last_update = getattr(props, "last_update_date", None) or "unknown"
            e1q, e2q, ero = [], [], []
            for g in getattr(props, "gates", []) or []:
                try:
                    err = props.gate_error(g.gate, g.qubits)
                except Exception:
                    continue
                if len(g.qubits) == 1:
                    e1q.append(err)
                elif len(g.qubits) == 2:
                    e2q.append(err)
            nq = getattr(self.hw_backend, "num_qubits", 0)
            for q in range(nq):
                try:
                    ero.append(props.readout_error(q))
                except Exception:
                    pass
            import numpy as _np
            if e1q:
                avg_1q = float(_np.mean(e1q))
            if e2q:
                avg_2q = float(_np.mean(e2q))
            if ero:
                avg_ro = float(_np.mean(ero))
        except Exception as e:
            print(f"[IBMSimulator] (note) avg-error fetch failed: {type(e).__name__}: {e}")

        def _fmt(v):
            return f"{v:.3e}" if isinstance(v, float) else "n/a"

        print(
            f"[IBMSimulator] Backend: AerSimulator(noise from {self.backend_name}, "
            f"calib_last_update={last_update}, avg_1Q={_fmt(avg_1q)}, "
            f"avg_2Q={_fmt(avg_2q)}, avg_readout={_fmt(avg_ro)})"
        )

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

    def _build_sub_coupling(self, used_physical_qubits):
        """`self.hw_backend.coupling_map`에서 `used_physical_qubits`에 속한 edge만
        남기고, 원본 물리 인덱스를 0-based 새 인덱스로 remap한 CouplingMap을 반환.

        예: hw coupling = [(0,1),(1,2),(5,6),(1,5)], used = [1,5,6]
            → sub_edges = [(0,1),(1,2)]  (1→0, 5→1, 6→2 — used 순서 그대로)

        주의: AerSimulator의 NoiseModel.from_backend는 원본 물리 인덱스 0..N_hw-1을
        키로 갖는다. remap 후 transpiled 회로의 새 qubit i는 AerSimulator 입장에서도
        qubit i로 노이즈가 적용되므로, 정확히는 `used_physical_qubits[i]`의 노이즈가
        아닌 물리 i의 노이즈가 입혀진다. 노이즈 패턴의 평균적 특성은 보존되지만
        spatial 정합성은 깨진다 (task scope상 노이즈 모델 자체 수정은 out of scope).
        """
        from qiskit.transpiler import CouplingMap
        idx_map = {phys: new for new, phys in enumerate(used_physical_qubits)}
        used_set = set(idx_map.keys())
        edges = self.hw_backend.coupling_map.get_edges()
        sub_edges = [
            [idx_map[a], idx_map[b]] for (a, b) in edges
            if a in used_set and b in used_set
        ]
        return CouplingMap(sub_edges)

    def transpile_circuit(self, circuit: QuantumCircuit,
                          initial_layout=None,
                          dd_sequence: str = None,
                          optimization_level: int = 2,
                          use_sub_coupling: bool = False) -> QuantumCircuit:
        """
        Transpile `circuit` to `self.hw_backend` via a preset pass manager with
        explicit Nighthawk basis gates + backend coupling map, optionally
        appending a `PadDynamicalDecoupling` scheduling pass.

        `self.hw_backend`는 simulator/qpu 분기 모두에서 실제 IBM 백엔드 객체이며,
        coupling_map/target/durations를 거기서 가져옴. AerSimulator(self.backend)는
        실행에만 사용.

        use_sub_coupling=True (simulator 분기 전용):
            initial_layout이 사용하는 물리 qubit subset의 sub-coupling만 추출해서
            transpile → 결과 회로 크기를 사용 qubit 수 정도로 축소 (AerSimulator
            메모리 폭발 방지). initial_layout 필수. dd_sequence는 sub-target 미작성
            때문에 강제 무시(경고만).
        use_sub_coupling=False (default, qpu 분기 호환):
            hw_backend.coupling_map 전체로 transpile.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        nighthawk_gates = ['cz', 'sx', 'rz', 'x', 'id']

        if use_sub_coupling:
            if initial_layout is None:
                raise ValueError(
                    "transpile_circuit(use_sub_coupling=True) requires initial_layout (got None)."
                )
            if dd_sequence:
                print(
                    f"[IBMSimulator] (warning) dd_sequence={dd_sequence!r} ignored under "
                    f"use_sub_coupling=True (sub-target not built)."
                )
                dd_sequence = None
            coupling_map_arg = self._build_sub_coupling(initial_layout)
            # 새 0-based 인덱스 공간으로 remap된 layout
            initial_layout_arg = list(range(len(initial_layout)))
        else:
            coupling_map_arg = self.hw_backend.coupling_map
            initial_layout_arg = initial_layout

        pm_transpile = generate_preset_pass_manager(
            optimization_level=optimization_level,
            basis_gates=nighthawk_gates,
            coupling_map=coupling_map_arg,
            initial_layout=initial_layout_arg,
        )
        transpiled = pm_transpile.run(circuit)
        if dd_sequence:
            from qiskit.transpiler import PassManager, InstructionDurations
            from qiskit.transpiler.passes import (
                ALAPScheduleAnalysis, PadDynamicalDecoupling,
            )
            from qiskit.circuit.library import XGate
            # ibm_miami basis = ['cz','sx','rz','x','id']; Y gate is not in
            # the basis so PadDynamicalDecoupling rejects Y-containing
            # sequences. Use X-only sequences instead.
            seqs = {
                "XX":  [XGate(), XGate()],
                "XX4": [XGate(), XGate(), XGate(), XGate()],
                "XX8": [XGate(), XGate(), XGate(), XGate(),
                        XGate(), XGate(), XGate(), XGate()],
            }
            if dd_sequence not in seqs:
                raise ValueError(f"Unknown DD sequence: {dd_sequence}")

            # ibm_miami preview target does not publish durations for reset
            # (and sometimes measure). Pull whatever is available from the
            # backend, then inject fallbacks only for instructions missing
            # from the collected table.
            try:
                durations = InstructionDurations.from_backend(self.hw_backend)
            except Exception:
                durations = InstructionDurations()

            fallback = [
                ("reset", None, 1000, "ns"),
                ("measure", None, 1000, "ns"),
                ("delay", None, 0, "ns"),
            ]
            for inst_name, qubits, val, unit in fallback:
                try:
                    durations.get(inst_name, qubits or 0)
                except Exception:
                    durations.update([(inst_name, qubits, val, unit)])

            pm = PassManager([
                ALAPScheduleAnalysis(
                    target=self.hw_backend.target,
                    durations=durations,
                ),
                PadDynamicalDecoupling(
                    target=self.hw_backend.target,
                    durations=durations,
                    dd_sequence=seqs[dd_sequence],
                    pulse_alignment=1,
                    skip_reset_qubits=True,
                ),
            ])
            transpiled = pm.run(transpiled)
        return transpiled

    def run(self, circuit: QuantumCircuit, shots: int = 1000,
            initial_layout=None, dd_sequence: str = None,
            optimization_level: int = 2,
            distance: int = None, num_rounds: int = None) -> dict:
        """
        회로를 IBM 백엔드에 제출하고 결과를 반환합니다.

        Args:
            circuit: 실행할 Qiskit 회로
            shots: 실행 횟수
            initial_layout: transpile 시 virtual->physical qubit layout
            dd_sequence: "XX", "XX4", "XX8" 또는 None (Y gate는 Nighthawk
                basis에 없어 지원하지 않음)
            optimization_level: transpile 최적화 레벨

        Returns:
            dict: {bitstring: count}
        """
        if self.backend_type == "simulator":
            # initial_layout이 가리키는 물리 qubit subset의 sub-coupling으로 transpile
            # 해서 회로를 사용 qubit 수만큼 축소 (AerSimulator 메모리 폭발 방지).
            # DD는 simulator 분기에서 항상 off (sub-target 미작성 회피).
            # capture는 simulator 분기에서 항상 비활성 (신드롬 박제 안 함).
            if dd_sequence:
                print(
                    f"[IBMSimulator] (warning) dd_sequence={dd_sequence!r} is ignored "
                    f"in simulator mode (DD off)."
                )
            transpiled = self.transpile_circuit(
                circuit,
                initial_layout=initial_layout,
                dd_sequence=None,
                optimization_level=optimization_level,
                use_sub_coupling=True,
            )
            print(
                f"[IBMSimulator] Simulating (shots={shots}, qubits={transpiled.num_qubits}, "
                f"depth={transpiled.depth()})"
            )
            job = self.backend.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts(transpiled)

        elif self.backend_type == "qpu":
            from qiskit_ibm_runtime import SamplerV2 as Sampler

            transpiled = self.transpile_circuit(
                circuit,
                initial_layout=initial_layout,
                dd_sequence=dd_sequence,
                optimization_level=optimization_level,
            )
            print(f"[IBMSimulator] Submitting (shots={shots}, qubits={transpiled.num_qubits}, "
                f"depth={transpiled.depth()})")

            sampler = Sampler(mode=self.backend)
            job = sampler.run([transpiled], shots=shots)
            print(f"[IBMSimulator] Job ID: {job.job_id()}")
            print(f"[IBMSimulator] Waiting for results...")
            result = job.result()

            # SamplerV2: register별 결과를 하나의 bitstring으로 합치기
            pub_result = result[0]
            creg_names = [creg.name for creg in circuit.cregs]
            print(f"[IBMSimulator] Classical registers: {creg_names}")

            # 각 register의 bitarray를 shot별로 합침
            from collections import Counter
            bitstrings = []
            num_shots = shots

            for i in range(num_shots):
                parts = []
                for name in creg_names:
                    reg_data = getattr(pub_result.data, name)
                    # BitArray에서 i번째 shot의 비트값을 문자열로
                    bits = reg_data.get_bitstrings()[i]
                    parts.append(bits)
                # Qiskit convention: "data_meas syn_r2 syn_r1 syn_r0"
                full_bitstring = " ".join(reversed(parts))
                bitstrings.append(full_bitstring)

            counts = dict(Counter(bitstrings))

            # Side-effect: hardware capture (capture=True 일 때만). 실패해도 counts 반환은 보장.
            if self.capture:
                try:
                    self._capture_qpu_run(
                        bitstrings=bitstrings,
                        pub_result=pub_result,
                        job=job,
                        circuit=circuit,
                        transpiled=transpiled,
                        shots=shots,
                        initial_layout=initial_layout,
                        dd_sequence=dd_sequence,
                        optimization_level=optimization_level,
                        distance=distance,
                        num_rounds=num_rounds,
                    )
                except Exception as e:
                    print(f"[IBMSimulator][capture] skipped (outer): {type(e).__name__}: {e}")

        else:
            # FakeBackend / Aer: 직접 실행
            job = self.backend.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts(transpiled)

        print(f"[IBMSimulator] Completed. Unique outcomes: {len(counts)}")
        return counts

    def _capture_qpu_run(self, *, bitstrings, pub_result, job, circuit, transpiled,
                          shots, initial_layout, dd_sequence, optimization_level,
                          distance, num_rounds):
        """
        QPU run 산출물을 시점별 디렉토리에 박제 (side-effect only).

        - 위치: PATHS.ibm_capture_dir(backend, distance, timestamp)
          (distance None이면 "unknown" → captures/{backend}/dunknown/{timestamp}/)
        - 산출물: per_shot.txt, pub_result.pkl(가능시), metadata.json, calibration.json
        - 각 항목 개별 try/except, metadata.json은 항상 시도하며 다른 항목 실패 사유를 안에 기록
        - 호출 측이 외부 try/except로 감싸므로 여기서 예외가 전파돼도 counts 반환은 무영향
        """
        import pickle
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d_for_path = int(distance) if distance is not None else "unknown"
        capture_dir = PATHS.ibm_capture_dir(self.backend_name, d_for_path, timestamp)
        print(f"[IBMSimulator][capture] -> {capture_dir}")

        errors = {}  # 다른 항목 실패 사유를 metadata에 기록

        # 1) per_shot.txt
        try:
            with open(os.path.join(capture_dir, "per_shot.txt"), "w") as f:
                f.write("\n".join(bitstrings))
        except Exception as e:
            errors["per_shot"] = f"{type(e).__name__}: {e}"

        # 2) pub_result.pkl (실패 허용)
        try:
            with open(os.path.join(capture_dir, "pub_result.pkl"), "wb") as f:
                pickle.dump(pub_result, f)
        except Exception as e:
            errors["pub_result_pickle"] = f"{type(e).__name__}: {e}"

        # 3) calibration.json (실패 허용; to_dict 부재 시 manual fallback)
        try:
            try:
                props = self.backend.properties()
            except Exception as e:
                props = None
                errors["calibration_fetch"] = f"{type(e).__name__}: {e}"
            if props is not None:
                cal_payload = None
                if hasattr(props, "to_dict"):
                    try:
                        cal_payload = props.to_dict()
                    except Exception as e:
                        errors["calibration_to_dict"] = f"{type(e).__name__}: {e}"
                if cal_payload is None:
                    # manual fallback: per-qubit + per-gate 핵심 값만
                    nq = getattr(self.backend, "num_qubits", 0)
                    qubits = []
                    for q in range(nq):
                        rec = {"qubit": q}
                        for attr in ("t1", "t2", "readout_error", "frequency"):
                            try:
                                rec[attr] = getattr(props, attr)(q)
                            except Exception:
                                pass
                        qubits.append(rec)
                    cal_payload = {"qubits": qubits, "_note": "manual fallback (to_dict 부재/실패)"}
                with open(os.path.join(capture_dir, "calibration.json"), "w") as f:
                    json.dump(cal_payload, f, default=str, indent=2)
        except Exception as e:
            errors["calibration"] = f"{type(e).__name__}: {e}"

        # 4) metadata.json (항상 시도)
        try:
            try:
                job_id = job.job_id()
            except Exception as e:
                job_id = None
                errors["job_id"] = f"{type(e).__name__}: {e}"
            try:
                backend_version = getattr(self.backend, "version", None) \
                    or getattr(self.backend, "backend_version", None)
            except Exception:
                backend_version = None
            try:
                cregs_info = [{"name": c.name, "size": c.size} for c in circuit.cregs]
            except Exception:
                cregs_info = None
            try:
                initial_layout_list = list(initial_layout) if initial_layout is not None else None
            except Exception:
                initial_layout_list = None

            meta = {
                "job_id": job_id,
                "timestamp": timestamp,
                "backend_name": self.backend_name,
                "backend_instance": self.backend_instance,
                "backend_type": self.backend_type,
                "backend_version": backend_version,
                "shots": shots,
                "distance": distance,
                "num_rounds": num_rounds,
                "initial_layout": initial_layout_list,
                "dd_sequence": dd_sequence,
                "optimization_level": optimization_level,
                "circuit": {
                    "num_qubits": circuit.num_qubits,
                    "num_clbits": circuit.num_clbits,
                    "depth": circuit.depth(),
                    "cregs": cregs_info,
                },
                "transpiled": {
                    "num_qubits": transpiled.num_qubits,
                    "depth": transpiled.depth(),
                },
                "capture_errors": errors or None,
            }
            with open(os.path.join(capture_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, default=str, indent=2)
        except Exception as e:
            # metadata도 실패하면 마지막으로 stderr 비슷하게 출력 (가벼운 single-file fallback)
            print(f"[IBMSimulator][capture] metadata.json write failed: {type(e).__name__}: {e}")

    def get_backend_info(self) -> dict:
        return {
            "backend_type": self.backend_type,
            "backend_name": self.backend_name,
            "num_qubits": getattr(self.backend, 'num_qubits', None),
        }


if __name__ == "__main__":
    print("=== IBM Simulator Test ===")

    # backend_name / backend_instance 는 config.json("backend" 섹션)을 따른다.
    _cfg_path = os.path.join(os.path.dirname(current_dir), "config.json")
    with open(_cfg_path) as _f:
        _bc = json.load(_f)["backend"]
    runner = IBMSimulator(
        backend_type="simulator",
        backend_instance=_bc["instance"],
        backend_name=_bc["backend_name"],
    )
    print(f"Backend info: {runner.get_backend_info()}")

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    counts = runner.run(qc, shots=100)
    print(f"Bell State: {counts}")
