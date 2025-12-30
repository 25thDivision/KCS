import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import networkx as nx
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
    """v25와 100% 동일한 색상 결정 로직"""
    val = int(round(2 * x) + round(2 * y))
    return val % 3

def plot_graph_view_final(distance=7):
    print(f">>> Generating Graph View Final (Synced with v25) (d={distance})...")
    
    # 1. 좌표 준비
    circuit = create_color_code_circuit(distance, distance, 0.0)
    qubit_coords = circuit.get_final_qubit_coordinates()
    detector_coords = circuit.get_detector_coordinates()
    
    q_indices = list(qubit_coords.keys())
    q_locs = np.array([qubit_coords[q] for q in q_indices])
    
    # 2. 에러 위치 선정 (v25와 동일)
    center_loc = np.mean(q_locs, axis=0)
    dists = np.linalg.norm(q_locs - center_loc, axis=1)
    center_q_idx = q_indices[np.argsort(dists)[0]] # X 에러
    z_error_q_idx = q_indices[np.argsort(dists)[7]] # Z 에러

    # 3. 강제 트리거 (거리 기반)
    x_triggered = set()
    qx, qy = qubit_coords[center_q_idx]
    for d_idx, coord in detector_coords.items():
        cx, cy = coord[0], -coord[1]
        if np.sqrt((cx - qx)**2 + ((-cy) - qy)**2) < 1.2:
            x_triggered.add(d_idx)
            
    z_triggered = set()
    qx, qy = qubit_coords[z_error_q_idx]
    for d_idx, coord in detector_coords.items():
        cx, cy = coord[0], -coord[1]
        if np.sqrt((cx - qx)**2 + ((-cy) - qy)**2) < 1.2:
            z_triggered.add(d_idx)

    # 4. 그래프 구성
    G = nx.Graph()
    det_keys = list(detector_coords.keys())
    G.add_nodes_from(det_keys)
    for i in range(len(det_keys)):
        for j in range(i + 1, len(det_keys)):
            u, v = det_keys[i], det_keys[j]
            coord_u = detector_coords[u]
            coord_v = detector_coords[v]
            if np.sqrt((coord_u[0]-coord_v[0])**2 + (coord_u[1]-coord_v[1])**2) < 2.5:
                G.add_edge(u, v)

    pos = {d: (detector_coords[d][0], -detector_coords[d][1]) for d in det_keys}

    # 5. 스타일 설정
    colors_x_on = ['#FF3333', '#00CC44', '#3366FF'] # RGB
    colors_z_on = ['#FF00FF', '#FFD700', '#00FFFF'] # CMY
    color_off = '#DDDDDD' 
    
    node_colors = []
    node_sizes = []
    node_edges = []
    
    for node in G.nodes():
        coord = detector_coords[node]
        # v25 로직으로 색상 타입 결정
        type_val = get_color_type_from_coord(coord[0], coord[1])
        
        is_x = node in x_triggered
        is_z = node in z_triggered
        
        if is_x:
            node_colors.append(colors_x_on[type_val])
            node_sizes.append(600)
            node_edges.append('black')
        elif is_z:
            node_colors.append(colors_z_on[type_val])
            node_sizes.append(600)
            node_edges.append('black')
        else:
            node_colors.append(color_off)
            node_sizes.append(200)
            node_edges.append('#AAAAAA')

    # 6. 그리기
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    nx.draw_networkx_edges(G, pos, width=2.0, edge_color='#888888', alpha=0.7, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                           edgecolors=node_edges, linewidths=2.0, ax=ax)

    ax.axis('off')
    
    # 범례
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_off, markeredgecolor='#AAAAAA', markersize=10, label='Off Node'),
        Line2D([0], [0], color='#888888', lw=2, label='Connectivity'),
        
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_x_on[0], markeredgecolor='black', markersize=12, label='X-Trig Red'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_z_on[0], markeredgecolor='black', markersize=12, label='Z-Trig Red'),
        
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_x_on[1], markeredgecolor='black', markersize=12, label='X-Trig Green'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_z_on[1], markeredgecolor='black', markersize=12, label='Z-Trig Green'),
        
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_x_on[2], markeredgecolor='black', markersize=12, label='X-Trig Blue'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_z_on[2], markeredgecolor='black', markersize=12, label='Z-Trig Blue')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fontsize=11, frameon=True, framealpha=1.0, edgecolor='#CCCCCC', ncol=4)

    plt.tight_layout()
    save_path = os.path.join(current_dir, f"view_graph_final_d{distance}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f">>> Graph View saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_graph_view_final(distance=7)