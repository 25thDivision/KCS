"""
Phase 2 디버깅 스크립트
 
IonQ에서 나온 실제 bitstring을 추적하여:
1. Qiskit bitstring 파싱이 올바른지 검증
2. 인코딩이 |0⟩_L을 정확히 만드는지 검증
3. No-error shot에서 신드롬이 000000인지 검증
4. Lookup Table이 올바르게 동작하는지 검증
"""
 
import os
import sys
import json
import numpy as np
 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
 
from circuits.qiskit_colorcode_generator import ColorCodeCircuit
from simulators.ionq_simulator import IonQSimulator
from extractors.syndrome_extractor import SyndromeExtractor
 
 
def load_config():
    config_path = os.path.join(current_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)
 
 
def debug_ideal_simulation():
    """
    Test 1: IDEAL(노이즈 없음) 시뮬레이션으로 기본 동작 검증
    노이즈가 없으면:
      - 모든 신드롬이 0이어야 함
      - 논리적 Z 값이 초기 상태(0)와 일치해야 함
    """
    print("=" * 60)
    print("  Test 1: Ideal Simulation (No Noise)")
    print("=" * 60)
    
    cc = ColorCodeCircuit(distance=3, num_rounds=1)
    qc = cc.build_circuit(initial_state=0)
    
    print(f"\n  Circuit: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"  Classical registers:")
    for creg in qc.cregs:
        print(f"    {creg.name}: {creg.size} bits")
    
    # Ideal 시뮬레이션 (노이즈 없음)
    runner = IonQSimulator(backend_type="simulator", noise_model="ideal")
    counts = runner.run(qc, shots=100)
    
    print(f"\n  Raw bitstrings (top 10):")
    for i, (bitstring, count) in enumerate(sorted(counts.items(), key=lambda x: -x[1])[:10]):
        print(f"    '{bitstring}' : {count}")
    
    # 파싱 테스트
    syn_indices = cc.get_syndrome_indices()
    extractor = SyndromeExtractor(syn_indices)
    
    print(f"\n  Parsing top bitstrings:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        syn_rounds, data_bits = extractor._parse_bitstring(bitstring)
        logical_z = extractor.compute_logical_value(data_bits)
        
        print(f"\n    Bitstring: '{bitstring}'")
        print(f"    Parsed syndrome (round 0): {syn_rounds[0]}")
        print(f"    Parsed data bits: {data_bits}")
        print(f"    Logical Z value: {logical_z}")
        print(f"    Count: {count}")
        
        # 검증
        all_syn_zero = np.all(syn_rounds == 0)
        if not all_syn_zero:
            print(f"    ⚠️ WARNING: Syndrome is not all-zero in ideal simulation!")
        if logical_z != 0:
            print(f"    ⚠️ WARNING: Logical Z != 0 for |0⟩_L encoding!")
 
 
def debug_noisy_simulation():
    """
    Test 2: Noisy 시뮬레이션에서 파싱 검증
    """
    print("\n" + "=" * 60)
    print("  Test 2: Noisy Simulation (forte-1)")
    print("=" * 60)
    
    config = load_config()
    backend_cfg = config["backend"]
    code_cfg = config["color_code"]
    
    distance = code_cfg["distances"][0]
    num_rounds = code_cfg["num_rounds_per_distance"][str(distance)]
    
    cc = ColorCodeCircuit(distance=distance, num_rounds=num_rounds)
    qc = cc.build_circuit(initial_state=0)
    
    print(f"\n  Circuit: d={distance}, rounds={num_rounds}")
    print(f"  Qubits: {qc.num_qubits}, Classical bits: {qc.num_clbits}")
    print(f"  Classical registers:")
    for creg in qc.cregs:
        print(f"    {creg.name}: {creg.size} bits")
    
    runner = IonQSimulator(
        backend_type=backend_cfg["type"],
        noise_model=backend_cfg["noise_model"]
    )
    counts = runner.run(qc, shots=100)
    
    syn_indices = cc.get_syndrome_indices()
    extractor = SyndromeExtractor(syn_indices)
    
    print(f"\n  Top 5 bitstrings:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        syn_rounds, data_bits = extractor._parse_bitstring(bitstring)
        logical_z = extractor.compute_logical_value(data_bits)
        
        print(f"\n    Bitstring: '{bitstring}'")
        print(f"    Parts: {bitstring.split(' ') if ' ' in bitstring else 'no spaces'}")
        for r in range(syn_rounds.shape[0]):
            print(f"    Syndrome round {r}: {syn_rounds[r]}")
        print(f"    Data bits: {data_bits}")
        print(f"    Logical Z (Z on q{syn_indices['logical_z']}): {logical_z}")
        print(f"    Count: {count}")
    
    # 전체 통계
    syndromes, data_states, shot_counts = extractor.extract_from_counts(counts)
    
    total = shot_counts.sum()
    all_zero_syn = np.all(syndromes == 0, axis=(1, 2))
    zero_syn_shots = shot_counts[all_zero_syn].sum()
    
    logical_values = np.array([extractor.compute_logical_value(d) for d in data_states])
    logical_0_shots = shot_counts[logical_values == 0].sum()
    logical_1_shots = shot_counts[logical_values == 1].sum()
    
    print(f"\n  === Statistics ===")
    print(f"  Total shots: {total}")
    print(f"  Zero-syndrome shots: {zero_syn_shots} ({zero_syn_shots/total*100:.1f}%)")
    print(f"  Logical 0: {logical_0_shots} ({logical_0_shots/total*100:.1f}%)")
    print(f"  Logical 1: {logical_1_shots} ({logical_1_shots/total*100:.1f}%)")
    
    # No Correction LER
    no_corr_errors = logical_1_shots  # |0⟩_L로 인코딩했으니 logical_Z=1이면 에러
    print(f"  No Correction LER: {no_corr_errors/total:.4f}")
 
 
def debug_lookup_table():
    """
    Test 3: Lookup Table이 단일 에러를 올바르게 정정하는지 검증
    """
    print("\n" + "=" * 60)
    print("  Test 3: Lookup Table Verification")
    print("=" * 60)
    
    from decoders.baseline_decoders import LookupTableDecoder, H_MATRIX
    
    lt = LookupTableDecoder()
    
    print(f"\n  H matrix:")
    print(f"  {H_MATRIX}")
    
    print(f"\n  Single-error syndrome table:")
    for q in range(7):
        syn = tuple(H_MATRIX[:, q].tolist())
        print(f"    q{q}: X-syndrome={syn}")
    
    # 단일 X-에러 시뮬레이션
    print(f"\n  Testing single X-error correction:")
    for err_q in range(7):
        # Z-syndrome이 X-에러를 감지
        z_syn = tuple(H_MATRIX[:, err_q].tolist())
        syn_vector = np.array([[list((0,0,0)) + list(z_syn)]], dtype=np.float32)
        
        # 데이터 상태: err_q가 flipped
        data = np.zeros((1, 7), dtype=np.int8)
        data[0, err_q] = 1
        
        correction = lt.decode(syn_vector, data)
        corrected = data[0] ^ correction[0]
        logical_z = (corrected[0] ^ corrected[1] ^ corrected[4])
        
        status = "✅" if logical_z == 0 else "❌"
        print(f"    X-error on q{err_q}: syn={z_syn}, correction={correction[0]}, "
              f"corrected={corrected}, logical_Z={logical_z} {status}")
 
 
def debug_bitstring_format():
    """
    Test 4: Qiskit bitstring 형식 확인
    """
    print("\n" + "=" * 60)
    print("  Test 4: Qiskit Bitstring Format")
    print("=" * 60)
    
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    
    # 간단한 회로로 Qiskit 비트 순서 확인
    data = QuantumRegister(3, "data")
    anc = QuantumRegister(2, "anc")
    c_syn = ClassicalRegister(2, "syn")
    c_data = ClassicalRegister(3, "data_meas")
    
    qc = QuantumCircuit(data, anc, c_syn, c_data)
    
    # data[0]만 1로 설정
    qc.x(data[0])
    # anc[1]만 1로 설정
    qc.x(anc[1])
    
    qc.measure(anc[0], c_syn[0])
    qc.measure(anc[1], c_syn[1])
    qc.measure(data[0], c_data[0])
    qc.measure(data[1], c_data[1])
    qc.measure(data[2], c_data[2])
    
    print(f"\n  Circuit: data[0]=1, anc[1]=1, rest=0")
    print(f"  Register order: syn(2 bits) then data_meas(3 bits)")
    print(f"  Expected: data[0]=1 → data_meas='001' (Qiskit reversed='001')")
    print(f"            anc[1]=1 → syn='10' (Qiskit reversed='10')")
    
    try:
        sim = AerSimulator()
        result = sim.run(qc, shots=10).result()
        counts = result.get_counts()
        print(f"\n  Aer result: {counts}")
        
        for bs in counts:
            print(f"    Bitstring: '{bs}'")
            if " " in bs:
                parts = bs.split(" ")
                print(f"    Parts: {parts}")
                print(f"    parts[0] (last register = data_meas): '{parts[0]}'")
                print(f"    parts[1] (first register = syn): '{parts[1]}'")
            else:
                print(f"    No spaces - single string")
    except ImportError:
        print("  ⚠️ qiskit-aer not installed. Skipping local test.")
        print("  Install with: pip install qiskit-aer")
 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, 
                       help="0=all, 1=ideal, 2=noisy, 3=lookup, 4=bitstring")
    args = parser.parse_args()
    
    if args.test == 0 or args.test == 4:
        debug_bitstring_format()
    if args.test == 0 or args.test == 3:
        debug_lookup_table()
    if args.test == 0 or args.test == 1:
        debug_ideal_simulation()
    if args.test == 0 or args.test == 2:
        debug_noisy_simulation()
    
    print("\n" + "=" * 60)
    print("  Debug Complete")
    print("=" * 60)
 