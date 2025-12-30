import numpy as np
import matplotlib.patches as patches

class SurfaceVisualizer:
    def __init__(self, distance):
        self.d = distance
        self.L = 2 * distance - 1
        
        # 격자 구조 재정의 (데이터셋의 좌표계와 동일)
        self.data_coords = []
        self.z_ancilla_coords = []
        self.x_ancilla_coords = []
        
        for r in range(self.L):
            for c in range(self.L):
                if (r + c) % 2 == 0:
                    self.data_coords.append((r,c))
                elif r % 2 == 0:
                    self.z_ancilla_coords.append((r,c))
                else:
                    self.x_ancilla_coords.append((r,c))

    def plot_sample(self, ax, syndrome_map, error_map, title=""):
        """
        syndrome_map: (2, L, L) - Input
        error_map: (2, L, L) - Label
        """
        # 1. 배경 격자 그리기
        # 연결선
        for r in range(self.L):
            for c in range(self.L):
                if c + 1 < self.L: ax.plot([c, c+1], [r, r], color='#dddddd', lw=1, zorder=0)
                if r + 1 < self.L: ax.plot([c, c], [r, r+1], color='#dddddd', lw=1, zorder=0)

        # 2. 노드 그리기 (비활성 상태)
        # Data Qubit (작은 회색 점)
        dy, dx = zip(*self.data_coords)
        ax.scatter(dx, dy, c='lightgray', s=50, zorder=1)
        
        # Z-Ancilla (초록 네모 - 비활성)
        zy, zx = zip(*self.z_ancilla_coords)
        ax.scatter(zx, zy, c='white', edgecolors='green', marker='s', s=100, alpha=0.3, zorder=1)

        # X-Ancilla (파랑 네모 - 비활성)
        xy, xx = zip(*self.x_ancilla_coords)
        ax.scatter(xx, xy, c='white', edgecolors='blue', marker='s', s=100, alpha=0.3, zorder=1)

        # 3. 데이터(Error) 그리기 (Label)
        # Channel 0: X Error (Red Star)
        x_err_rows, x_err_cols = np.where(error_map[0] == 1)
        if len(x_err_rows) > 0:
            ax.scatter(x_err_cols, x_err_rows, c='red', marker='*', s=200, label='X-Error', zorder=3)
            
        # Channel 1: Z Error (Blue Star)
        z_err_rows, z_err_cols = np.where(error_map[1] == 1)
        if len(z_err_rows) > 0:
            ax.scatter(z_err_cols, z_err_rows, c='blue', marker='*', s=200, label='Z-Error', zorder=3)

        # 4. 증후군(Syndrome) 그리기 (Input)
        # Channel 0: Z-Syndrome (Orange Circle around Z-Ancilla) -> Detects X Error
        z_syn_rows, z_syn_cols = np.where(syndrome_map[0] == 1)
        for r, c in zip(z_syn_rows, z_syn_cols):
            # Z-Ancilla 위치인지 확인 (Safety check)
            ax.add_patch(patches.Circle((c, r), 0.35, linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.5, label='Z-Syndrome (Detects X)', zorder=2))

        # Channel 1: X-Syndrome (Purple Circle around X-Ancilla) -> Detects Z Error
        x_syn_rows, x_syn_cols = np.where(syndrome_map[1] == 1)
        for r, c in zip(x_syn_rows, x_syn_cols):
            ax.add_patch(patches.Circle((c, r), 0.35, linewidth=2, edgecolor='purple', facecolor='purple', alpha=0.5, label='X-Syndrome (Detects Z)', zorder=2))

        # 스타일링
        ax.set_title(title, fontsize=12, pad=10)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 범례 (중복 제거)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)