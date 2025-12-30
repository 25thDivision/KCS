import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from matplotlib.colors import to_rgb

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from generators.color_code import create_color_code_circuit
except ImportError:
    from simulation.generators.color_code import create_color_code_circuit

def get_color_type_from_coord(x, y):
    """v25와 100% 동일한 색상 결정 로직"""
    val = int(round(2 * x) + round(2 * y))
    return val % 3

def plot_image_view_final(distance=7):
    print(f">>> Generating Image View Final (Synced with v25) (d={distance})...")
    
    # 1. 좌표 준비
    circuit = create_color_code_circuit(distance, distance, 0.0)
    qubit_coords = circuit.get_final_qubit_coordinates()
    detector_coords = circuit.get_detector_coordinates()
    
    q_indices = list(qubit_coords.keys())
    q_locs = np.array([qubit_coords[q] for q in q_indices])
    
    # 2. 에러 위치 선정
    center_loc = np.mean(q_locs, axis=0)
    dists = np.linalg.norm(q_locs - center_loc, axis=1)
    center_q_idx = q_indices[np.argsort(dists)[0]]
    z_error_q_idx = q_indices[np.argsort(dists)[7]]

    # 3. 강제 트리거 (좌표와 함께 저장)
    x_triggered_coords = [] # list of (x, y)
    qx, qy = qubit_coords[center_q_idx]
    for _, coord in detector_coords.items():
        cx, cy = coord[0], -coord[1]
        if np.sqrt((cx - qx)**2 + ((-cy) - qy)**2) < 1.2:
            x_triggered_coords.append((coord[0], coord[1]))
            
    z_triggered_coords = []
    qx, qy = qubit_coords[z_error_q_idx]
    for _, coord in detector_coords.items():
        cx, cy = coord[0], -coord[1]
        if np.sqrt((cx - qx)**2 + ((-cy) - qy)**2) < 1.2:
            z_triggered_coords.append((coord[0], coord[1]))

    # 4. 이미지 생성
    xs = [c[0] for c in detector_coords.values()]
    ys = [c[1] for c in detector_coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = int(max_x - min_x) + 3
    height = int(max_y - min_y) + 3
    
    # 배경
    bg_color = to_rgb('#F5F5F5')
    grid_img = np.full((height, width, 3), bg_color)
    
    # 팔레트 (v25와 동일)
    colors_x_on = [to_rgb(c) for c in ['#FF3333', '#00CC44', '#3366FF']]
    colors_z_on = [to_rgb(c) for c in ['#FF00FF', '#FFD700', '#00FFFF']]
    
    # 5. 픽셀 칠하기
    # X 에러
    for (tx, ty) in x_triggered_coords:
        r = int(ty - min_y) + 1
        c = int(tx - min_x) + 1
        if 0 <= r < height and 0 <= c < width:
            type_val = get_color_type_from_coord(tx, ty)
            grid_img[r, c] = colors_x_on[type_val] # 해당 타입의 색상 적용
            
    # Z 에러
    for (tx, ty) in z_triggered_coords:
        r = int(ty - min_y) + 1
        c = int(tx - min_x) + 1
        if 0 <= r < height and 0 <= c < width:
            type_val = get_color_type_from_coord(tx, ty)
            grid_img[r, c] = colors_z_on[type_val] # 해당 타입의 색상 적용

    # 6. 그리기
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_img, origin='lower', interpolation='nearest')
    
    # 그리드 및 스타일
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    ax.set_xlabel("Column Index", fontsize=12)
    ax.set_ylabel("Row Index", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(current_dir, f"view_image_final_d{distance}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f">>> Image View saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_image_view_final(distance=7)