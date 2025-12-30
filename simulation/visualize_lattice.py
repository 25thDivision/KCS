import matplotlib.pyplot as plt
import numpy as np
import stim
import sys
import os
from matplotlib.patches import RegularPolygon

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
# visualize_lattice.py가 simulation 폴더 안에 있다고 가정할 때의 상위 경로 설정
# 만약 Import 에러가 나면 이 부분을 조정해야 합니다.
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from generators.color_code import create_color_code_circuit
except ImportError:
    # 경로가 안 맞을 경우를 대비해 simulation.generators로 시도
    from simulation.generators.color_code import create_color_code_circuit

def plot_physical_lattice(distance=5):
    print(f">>> Generating Physical Lattice Visualization (d={distance})...")
    
    # 1. 회로 생성
    circuit = create_color_code_circuit(distance, distance, 0.001)
    
    # 2. 좌표 데이터 추출
    # (1) 큐비트 좌표 (검은 점)
    qubit_coords = circuit.get_final_qubit_coordinates()
    qx = [v[0] for v in qubit_coords.values()]
    qy = [-v[1] for v in qubit_coords.values()] # Y축 반전

    # (2) 디텍터(스태빌라이저) 좌표 (색칠된 면)
    detector_coords = circuit.get_detector_coordinates()
    
    # 3. 색상 분류 (좌표 기반으로 R, G, B 추정)
    red_coords, green_coords, blue_coords = [], [], []
    
    for idx, coord in detector_coords.items():
        x, y = coord[0], -coord[1]
        
        # Stim의 Color Code 좌표 규칙성 활용
        if (int(coord[0]) + int(coord[1])) % 3 == 0:
            red_coords.append((x, y))
        elif (int(coord[0]) + int(coord[1])) % 3 == 1:
            green_coords.append((x, y))
        else:
            blue_coords.append((x, y))

    # 4. 그리기
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # (A) 스태빌라이저 그리기
    marker_size = 2200 // distance 
    
    ax.scatter([p[0] for p in red_coords], [p[1] for p in red_coords], 
               c='#FF6B6B', s=marker_size, marker='h', label='Red Stabilizers', alpha=0.9, edgecolors='none')
    ax.scatter([p[0] for p in green_coords], [p[1] for p in green_coords], 
               c='#4ECDC4', s=marker_size, marker='h', label='Green Stabilizers', alpha=0.9, edgecolors='none')
    ax.scatter([p[0] for p in blue_coords], [p[1] for p in blue_coords], 
               c='#4D96FF', s=marker_size, marker='h', label='Blue Stabilizers', alpha=0.9, edgecolors='none')

    # (B) 큐비트 그리기
    ax.scatter(qx, qy, c='black', s=80, zorder=10, label='Data Qubits')

    # 타이틀 및 스타일
    plt.title(f"Physical Lattice Layout (Color Code d={distance})", fontsize=20, fontweight='bold', pad=20)
    plt.axis('off')
    
    # 범례 설정 [수정된 부분]
    lgnd = plt.legend(loc='upper right', fontsize=12, frameon=True)
    # legendHandles -> legend_handles 로 수정
    for handle in lgnd.legend_handles:
        handle.set_sizes([100]) 

    plt.tight_layout()
    
    # 파일 저장
    save_path = os.path.join(current_dir, f"physical_lattice_d{distance}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f">>> Figure saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_physical_lattice(distance=5)