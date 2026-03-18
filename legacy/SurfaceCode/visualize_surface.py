import numpy as np
import matplotlib.pyplot as plt
import os
from generators.surface_visualizer import SurfaceVisualizer

# --- 설정 ---
D_TARGET = 5
P_TARGET = 0.01

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "../dataset/surface_code")
FILE_PATH = f"{DATA_DIR}/surface_test_d{D_TARGET}_p{P_TARGET}.npz"
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_interesting_sample(X, y, error_type='X'):
    """
    데이터셋에서 설명하기 좋은 예제를 찾습니다.
    조건 1: 지정한 타입(X or Z)의 에러만 있을 것.
    조건 2: 에러 개수가 1~3개로 적당할 것 (너무 많으면 난잡함).
    조건 3: [NEW] 에러가 격자 '내부(Bulk)'에 위치할 것 (가장자리 제외).
    """
    N, _, L, _ = y.shape
    center_min = 1      # 가장자리(0번 인덱스) 제외
    center_max = L - 2  # 가장자리(마지막 인덱스) 제외
    
    # 만약 격자가 너무 작으면(d=3) 내부 조건 완화
    if L <= 3:
        center_min, center_max = 0, L - 1

    candidate_idx = 0
    
    for i in range(N):
        # 1. 에러 타입 확인
        has_x_err = np.any(y[i, 0] == 1)
        has_z_err = np.any(y[i, 1] == 1)
        
        # 2. 에러 위치(좌표) 가져오기
        if error_type == 'X':
            if not has_x_err or has_z_err: continue
            rows, cols = np.where(y[i, 0] == 1)
        else:
            if not has_z_err or has_x_err: continue
            rows, cols = np.where(y[i, 1] == 1)
            
        # 3. 에러 개수 체크 (2~4개)
        if not (1 < len(rows) <= 4):
            continue

        # 4. [핵심] 위치가 내부에 있는지 확인
        # 모든 에러가 안전 구역(center_min ~ center_max) 안에 있어야 함
        is_inner = np.all((rows >= center_min) & (rows <= center_max) & 
                          (cols >= center_min) & (cols <= center_max))
        
        if is_inner:
            return i  # 딱 좋은 샘플 발견! 즉시 반환
            
        # 차선책: 내부 조건은 만족 못 했지만, 타입과 개수는 맞는 거라도 일단 기억해둠
        candidate_idx = i
            
    # 완벽한 내부 샘플을 못 찾았으면 차선책 반환
    return candidate_idx

# __main__
if not os.path.exists(FILE_PATH):
    print(f"파일을 찾을 수 없습니다: {FILE_PATH}")
    print("dataset 폴더에 해당 파일이 있는지 확인하세요.")
else:
    print(f"Dataset 로드 중... {FILE_PATH}")
    data = np.load(FILE_PATH)
    X_data = data['X'] # (N, 2, L, L)
    y_data = data['y'] # (N, 2, L, L)
    
    viz = SurfaceVisualizer(D_TARGET)
    
    # 1. Figure 1: Surface Code 구조 (빈 격자)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    empty_map = np.zeros((2, viz.L, viz.L))
    viz.plot_sample(ax1, empty_map, empty_map, title=f"Surface Code Structure (d={D_TARGET})")
    
    # 범례용 가짜 점 찍기
    ax1.scatter([], [], c='lightgray', s=50, label='Data Qubit')
    ax1.scatter([], [], c='white', edgecolors='green', marker='s', label='Z-Ancilla')
    ax1.scatter([], [], c='white', edgecolors='blue', marker='s', label='X-Ancilla')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_structure.png", dpi=300, bbox_inches='tight')
    print("Figure 1 저장 완료.")

    # 2. Figure 2: X-Error 메커니즘 (실제 샘플)
    idx_x = find_interesting_sample(X_data, y_data, 'X')
    print(f"X 에러 예제 샘플 인덱스: {idx_x}")
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    viz.plot_sample(ax2, X_data[idx_x], y_data[idx_x], title=f"Real Sample: X-Error Mechanism (Index {idx_x})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_x_mechanism.png", dpi=300, bbox_inches='tight')
    print("Figure 2 저장 완료.")

    # 3. Figure 3: Z-Error 메커니즘 (실제 샘플)
    idx_z = find_interesting_sample(X_data, y_data, 'Z')
    print(f"Z 에러 예제 샘플 인덱스: {idx_z}")
    
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    viz.plot_sample(ax3, X_data[idx_z], y_data[idx_z], title=f"Real Sample: Z-Error Mechanism (Index {idx_z})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_z_mechanism.png", dpi=300, bbox_inches='tight')
    print("Figure 3 저장 완료.")
    
    plt.show()