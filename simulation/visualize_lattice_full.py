import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
from collections import defaultdict

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from generators.color_code import create_color_code_circuit
except ImportError:
    from simulation.generators.color_code import create_color_code_circuit

def get_color_type_from_coord(x, y):
    """
    [핵심 솔루션] 좌표 기반 3-Coloring 공식
    사용자 데이터(3.5 등)를 고려하여 2배 스케일링 후 나머지 연산.
    이 식은 육각형 격자에서 인접한 면들이 서로 다른 값을 갖도록 보장합니다.
    """
    # 2배를 곱해 0.5 단위를 정수로 만듦
    val = int(round(2 * x) + round(2 * y))
    return val % 3

def plot_poster_final_v25(distance=7):
    print(f">>> Generating Poster Figure V25 (Geometry-Based Trigger + Perfect Coloring) (d={distance})...")
    
    # 1. 회로 및 좌표 준비
    circuit = create_color_code_circuit(distance, distance, 0.0)
    qubit_coords = circuit.get_final_qubit_coordinates()
    detector_coords = circuit.get_detector_coordinates()
    
    q_indices = list(qubit_coords.keys())
    q_locs = np.array([qubit_coords[q] for q in q_indices])
    
    # 2. 에러 위치 선정
    center_loc = np.mean(q_locs, axis=0)
    dists = np.linalg.norm(q_locs - center_loc, axis=1)
    
    center_q_idx = q_indices[np.argsort(dists)[0]] # X 에러 (중앙)
    z_error_q_idx = q_indices[np.argsort(dists)[7]] # Z 에러 (내부 안전지대)

    # 3. 강제 트리거 (Geometry 방식 - 무조건 작동함)
    x_triggered_pos = []
    qx, qy = qubit_coords[center_q_idx]
    for _, coord in detector_coords.items():
        cx, cy = coord[0], -coord[1]
        # 에러 큐비트와 면 중심 거리가 가까우면(1.2) 무조건 켜진 것으로 간주
        if np.sqrt((cx - qx)**2 + ((-cy) - qy)**2) < 1.2: 
            x_triggered_pos.append((round(coord[0], 3), round(coord[1], 3)))
            
    z_triggered_pos = []
    qx, qy = qubit_coords[z_error_q_idx]
    for _, coord in detector_coords.items():
        cx, cy = coord[0], -coord[1]
        if np.sqrt((cx - qx)**2 + ((-cy) - qy)**2) < 1.2:
            z_triggered_pos.append((round(coord[0], 3), round(coord[1], 3)))

    # 4. 좌표별 디텍터 묶기 (중복 좌표 제거용)
    detectors_by_pos = defaultdict(list)
    for d_idx, coord in detector_coords.items():
        pos_key = (round(coord[0], 3), round(coord[1], 3))
        detectors_by_pos[pos_key].append(d_idx)

    pos_keys = list(detectors_by_pos.keys())

    # 5. 그리기
    fig, ax = plt.subplots(figsize=(12, 14)) 
    ax.set_aspect('equal')
    
    # [색상 정의]
    # X 반응 (RGB) - v19/GraphView 통일
    colors_x_on = ['#FF3333', '#00CC44', '#3366FF'] 
    # Z 반응 (CMY)
    colors_z_on = ['#FF00FF', '#FFD700', '#00FFFF'] 
    # 꺼진 면 (v5/v19 파스텔톤)
    colors_off = ['#FF6B6B', '#76D7C4', '#5DADE2'] 
    
    for idx, pos_key in enumerate(pos_keys):
        cx, cy = pos_key[0], -pos_key[1]
        
        # [핵심] 수학적 컬러링 타입 결정 (0, 1, 2)
        color_idx = get_color_type_from_coord(cx, -cy)
        
        is_x_triggered = pos_key in x_triggered_pos
        is_z_triggered = pos_key in z_triggered_pos
        
        # 다각형 좌표 구하기
        neighbor_qubits = []
        for q_idx, q_coord in qubit_coords.items():
            qx, qy = q_coord[0], -q_coord[1]
            if np.sqrt((cx - qx)**2 + (cy - qy)**2) < 1.4:
                neighbor_qubits.append([qx, qy])
        
        if len(neighbor_qubits) < 3: continue
        
        points = np.array(neighbor_qubits)
        hull = ConvexHull(points)
        polygon_verts = points[hull.vertices]
        
        # [스타일 결정]
        if is_x_triggered:
            # X 에러: 해당 타입의 RGB 색상
            face_color = colors_x_on[color_idx]
            edge_color = 'black'
            alpha = 1.0
            z_order = 10
            line_width = 3.0
        elif is_z_triggered:
            # Z 에러: 해당 타입의 CMY 색상
            face_color = colors_z_on[color_idx]
            edge_color = 'black'
            alpha = 1.0
            z_order = 10
            line_width = 3.0
        else:
            # 꺼진 면: 해당 타입의 파스텔톤 (v19 스타일)
            face_color = colors_off[color_idx]
            edge_color = 'white' 
            alpha = 0.5 # 반투명해서 예쁘게
            z_order = 1
            line_width = 1.5
        
        poly = Polygon(polygon_verts, closed=True, facecolor=face_color, edgecolor=edge_color, linewidth=line_width, alpha=alpha, zorder=z_order)
        ax.add_patch(poly)

    # 6. 큐비트
    qx_list = [v[0] for v in qubit_coords.values()]
    qy_list = [-v[1] for v in qubit_coords.values()]
    ax.scatter(qx_list, qy_list, c='white', s=100, edgecolors='#555555', linewidths=1.2, zorder=15)

    # 7. 에러 마커
    cx, cy = qubit_coords[center_q_idx]
    ax.scatter(cx, -cy, c='black', s=600, marker='x', linewidths=5, zorder=20)
    tx, ty = qubit_coords[z_error_q_idx]
    ax.scatter(tx, -ty, c='black', s=600, marker='+', linewidths=5, zorder=20)

    # 8. 범례
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', markeredgecolor='black', markersize=15, markeredgewidth=3, label='X Error (triggers Z-stabs)'),
        Line2D([0], [0], marker='+', color='w', markeredgecolor='black', markersize=15, markeredgewidth=3, label='Z Error (triggers X-stabs)'),
        
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_x_on[0], markersize=12, label='X-Triggered Red'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_z_on[0], markersize=12, label='Z-Triggered Red'),
        
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_x_on[1], markersize=12, label='X-Triggered Green'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_z_on[1], markersize=12, label='Z-Triggered Green'),
        
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_x_on[2], markersize=12, label='X-Triggered Blue'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_z_on[2], markersize=12, label='Z-Triggered Blue')
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fontsize=12, frameon=True, framealpha=1.0, edgecolor='#CCCCCC', ncol=2)

    plt.axis('off')
    plt.tight_layout()
    
    save_path = os.path.join(current_dir, f"poster_figure_final_d{distance}_v25.png")
    # 투명 배경으로 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f">>> Figure saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_poster_final_v25(distance=7)