import stim
import numpy as np

# 경로 문제 해결을 위한 임시 코드 (필요시)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 만약 경로 에러가 나면, simulation 폴더 안에서 실행하거나 경로를 맞춰주세요.
try:
    from generators.color_code import create_color_code_circuit
except:
    from simulation.generators.color_code import create_color_code_circuit

def check_coordinates(distance=7):
    circuit = create_color_code_circuit(distance, distance, 0.0)
    detector_coords = circuit.get_detector_coordinates()
    
    print("\n=== Stim Detector Coordinates (First 20) ===")
    # y좌표 기준으로 정렬해서 출력
    sorted_dets = sorted(detector_coords.items(), key=lambda item: (-item[1][1], item[1][0]))
    
    for i, (d_idx, coord) in enumerate(sorted_dets[:20]):
        print(f"Detector {d_idx}: (x={coord[0]:.1f}, y={coord[1]:.1f})")
        
    print("\n=== Center & Top Qubit Coordinates ===")
    qubit_coords = circuit.get_final_qubit_coordinates()
    q_locs = np.array(list(qubit_coords.values()))
    center_loc = np.mean(q_locs, axis=0)
    print(f"Center Estim: {center_loc}")

if __name__ == "__main__":
    check_coordinates(7)