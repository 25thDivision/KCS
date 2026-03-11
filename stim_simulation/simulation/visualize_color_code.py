import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

# 육각형 중심점 좌표 및 색상 할당
sq3 = np.sqrt(3)
centers = {
    'c0': (0, 0, 'R'),
    'c1': (1.5, sq3/2, 'G'),
    'c5': (0, sq3, 'B'),
    'c4': (-1.5, sq3/2, 'G'),
    'c3': (-1.5, -sq3/2, 'B'),
    'c6': (0, -sq3, 'G'),
    'c2': (1.5, -sq3/2, 'B'),
}

# 파스텔 톤 배경색과 불이 켜졌을 때의 강조색
light_colors = {'R': '#ffcccc', 'G': '#ccffcc', 'B': '#ccccff'}
active_colors = {'R': '#ff3333', 'G': '#33cc33', 'B': '#3366ff'}

# 에러가 발생한 꼭짓점 (물리적 큐비트 위치)
v1 = (0.5, sq3/2)
v2 = (1.0, 0)

# 1. 그림 크기를 넉넉하게 키움 (16, 8)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, case in zip(axes, ['single', 'double']):
    ax.set_aspect('equal')
    ax.axis('off')
    
    if case == 'single':
        active_faces = ['c0', 'c1', 'c5']  # 3개의 면 점등
        errors = [v1]
        ax.set_title("Single Qubit Error\n(3 Faces Light Up)", fontsize=25, pad=25)
    else:
        # 공통 면(c0, c1)은 상쇄되어 꺼지고, 양 끝 면(c5, c2)만 남음
        active_faces = ['c5', 'c2'] 
        errors = [v1, v2]
        ax.set_title("Two Adjacent Errors\n(Shared Faces Cancel Out)", fontsize=25, pad=25)

    # 격자 그리기
    for key, (x, y, color) in centers.items():
        is_active = key in active_faces
        facecolor = active_colors[color] if is_active else light_colors[color]
        edgecolor = '#333333' if is_active else '#999999'
        linewidth = 3 if is_active else 1
        
        hex_patch = RegularPolygon((x, y), numVertices=6, radius=1, 
                                   orientation=np.pi/6, facecolor=facecolor, 
                                   edgecolor=edgecolor, linewidth=linewidth, alpha=0.9)
        ax.add_patch(hex_patch)
        
        if is_active:
            ax.text(x, y, "Syndrome!", ha='center', va='center', fontsize=20, fontweight='bold', color='white')
            
    # 에러 및 에러 선(String) 그리기
    for idx, (ex, ey) in enumerate(errors):
        ax.plot(ex, ey, marker='X', color='red', markersize=16, markeredgecolor='black', markeredgewidth=2)
        # 2. 에러 텍스트 위치를 마커와 안 겹치게 조금 더 벌림
        ax.text(ex + 0.2, ey + 0.2, f"Error {idx+1}", color='black', fontsize=15, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        
    if case == 'double':
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color='red', linewidth=5, linestyle='--')

    # 3. 그림이 잘리지 않도록 축(xlim, ylim)의 상하좌우 여백을 더 넓게 설정
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)

# 4. 저장 시 bbox_inches='tight' 옵션을 추가하여 모든 텍스트가 잘리지 않고 온전히 포함되도록 강제
plt.tight_layout()
plt.savefig('color_code_errors_fixed.png', dpi=300, bbox_inches='tight', pad_inches=0.3)