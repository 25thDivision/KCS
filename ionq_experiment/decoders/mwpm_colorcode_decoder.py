"""
Color Code용 MWPM 디코더 (Restriction Decoder)

Color code에서는 각 데이터 큐빗이 최대 3개의 stabilizer에 참여하므로
standard MWPM을 직접 적용할 수 없습니다 (Lacroix et al., Nature 2025).

대신 Restriction Decoder (Delfosse, 2014)를 사용합니다:
  1. Color code를 3개의 색상 쌍 (RG, RB, GB)으로 제한 (restrict)
  2. 각 제한은 surface-code-like 구조 → PyMatching으로 MWPM 실행
  3. Full syndrome consistency check: H·correction = syndrome (mod 2)
  4. Consistent corrections 중 minimum weight 선택

Steane Code [[7,1,3]] 특성:
  H 행렬의 7개 열이 모두 distinct → 모든 단일 에러 신드롬이 고유
  따라서 d=3에서 MWPM = Lookup Table (이론적으로 동일한 결과)
  
  이 구현은 restriction MWPM + consistency check를 사용하되,
  모든 restriction이 inconsistent한 edge case에서는 direct syndrome lookup으로 fallback.
  d>3 확장 시 restriction decoder가 본래의 역할을 합니다.

Parity Check Matrix:
  H = [[1,1,1,1,0,0,0],  # R (Red)
       [0,1,1,0,1,1,0],  # G (Green)
       [0,0,1,1,0,1,1]]  # B (Blue)

Restrictions:
  RG: R과 G만 → 공유{1,2}, R경계{0,3}, G경계{4,5}
  RB: R과 B만 → 공유{2,3}, R경계{0,1}, B경계{5,6}  
  GB: G와 B만 → 공유{2,5}, G경계{1,4}, B경계{3,6}
"""

import numpy as np

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False
    print("[WARNING] pymatching not installed. MWPMColorCodeDecoder will use fallback.")

from scipy.sparse import csc_matrix


# ==============================================================================
# Steane Code [[7,1,3]] Parity Check Matrix
# ==============================================================================
H_MATRIX = np.array([
    [1, 1, 1, 1, 0, 0, 0],  # R (Red)
    [0, 1, 1, 0, 1, 1, 0],  # G (Green)
    [0, 0, 1, 1, 0, 1, 1],  # B (Blue)
], dtype=np.uint8)

# 3개의 색상 쌍 제한
RESTRICTIONS = {
    "RG": (0, 1),  # Red, Green
    "RB": (0, 2),  # Red, Blue
    "GB": (1, 2),  # Green, Blue
}

# 단일 에러 신드롬 → 큐빗 인덱스 (모든 7개 열이 distinct)
SINGLE_ERROR_TABLE = {}
for q in range(7):
    syndrome = tuple(H_MATRIX[:, q].tolist())
    SINGLE_ERROR_TABLE[syndrome] = q


class MWPMColorCodeDecoder:
    """
    Color Code [[7,1,3]] 전용 MWPM Restriction Decoder.
    
    알고리즘:
      1. 3개의 restriction (RG, RB, GB) 각각에서 MWPM 실행
      2. 각 correction의 full syndrome consistency 검증: H·c = s (mod 2)
      3. Consistent corrections 중 minimum weight 선택
      4. 모든 restriction이 inconsistent → direct syndrome lookup fallback
    
    인터페이스: decode(syndromes, data_states) → corrections
    """
    
    def __init__(self, weights: np.ndarray = None):
        """
        Args:
            weights: (7,) 각 큐빗의 에러 가중치 (None이면 uniform weight=1)
        """
        self.num_data = 7
        self.h_matrix = H_MATRIX
        self.weights = weights
        
        # 3개 restriction별 PyMatching matcher 초기화
        self.matchers = {}
        self.restricted_h = {}
        
        if HAS_PYMATCHING:
            for name, (row_i, row_j) in RESTRICTIONS.items():
                h_restricted = self.h_matrix[[row_i, row_j], :]
                self.restricted_h[name] = h_restricted
                
                h_sparse = csc_matrix(h_restricted)
                if weights is not None:
                    self.matchers[name] = pymatching.Matching(h_sparse, weights=weights)
                else:
                    self.matchers[name] = pymatching.Matching(h_sparse)
        
        # 미리 모든 3-bit syndrome에 대한 correction 캐싱
        self._build_correction_cache()
    
    def _build_correction_cache(self):
        """
        모든 가능한 3-bit syndrome (8가지)에 대해 correction을 미리 계산합니다.
        d=3에서는 이 캐시로 O(1) lookup이 가능합니다.
        """
        self._cache = {}
        
        for s0 in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    syn = np.array([s0, s1, s2], dtype=np.uint8)
                    self._cache[(s0, s1, s2)] = self._compute_correction(syn)
    
    def _compute_correction(self, syndrome_3bit: np.ndarray) -> np.ndarray:
        """
        3-bit syndrome에 대한 MWPM correction을 계산합니다.
        
        Args:
            syndrome_3bit: (3,) — [R, G, B] 신드롬
        
        Returns:
            correction: (7,) — correction 벡터
        """
        # Zero syndrome → no correction
        if syndrome_3bit.sum() == 0:
            return np.zeros(7, dtype=np.int8)
        
        if not HAS_PYMATCHING:
            return self._direct_lookup(syndrome_3bit)
        
        # === Restriction MWPM + Full Syndrome Consistency Check ===
        candidates = []
        
        for name, (row_i, row_j) in RESTRICTIONS.items():
            restricted_syn = np.array([syndrome_3bit[row_i], syndrome_3bit[row_j]], dtype=np.uint8)
            
            # 이 restriction에서 defect가 없으면 skip
            if restricted_syn.sum() == 0:
                continue
            
            # MWPM 디코딩
            correction = self.matchers[name].decode(restricted_syn)
            
            # ★ Full syndrome consistency check: H·c = syndrome (mod 2)
            predicted_syn = (self.h_matrix @ correction) % 2
            if np.array_equal(predicted_syn, syndrome_3bit):
                candidates.append((int(correction.sum()), correction.astype(np.int8)))
        
        if candidates:
            # Consistent corrections 중 minimum weight 선택
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        
        # === Fallback: Direct syndrome lookup ===
        # 모든 restriction이 inconsistent한 edge case (d=3에서 q3, q6 등)
        # Direct lookup은 MWPM과 동일 결과 (d=3, uniform noise)
        return self._direct_lookup(syndrome_3bit)
    
    def _direct_lookup(self, syndrome_3bit: np.ndarray) -> np.ndarray:
        """
        Syndrome → 단일 큐빗 에러 direct lookup.
        
        d=3 Steane code에서 H의 모든 열이 distinct하므로,
        모든 non-zero syndrome이 고유한 단일 큐빗 에러에 대응합니다.
        이는 uniform noise 하에서 MWPM이 내리는 결정과 동일합니다.
        """
        correction = np.zeros(7, dtype=np.int8)
        syn_tuple = tuple(syndrome_3bit.astype(int).tolist())
        error_qubit = SINGLE_ERROR_TABLE.get(syn_tuple)
        if error_qubit is not None:
            correction[error_qubit] = 1
        return correction
    
    def decode(self, syndromes: np.ndarray, data_states: np.ndarray) -> np.ndarray:
        """
        MWPM Restriction Decoder로 correction을 계산합니다.
        
        Args:
            syndromes: (N, num_rounds, 6) 또는 (N, 6)
                       순서: [X-Red, X-Green, X-Blue, Z-Red, Z-Green, Z-Blue]
            data_states: (N, 7)
        
        Returns:
            corrections: (N, 7)
        """
        # 마지막 라운드의 신드롬만 사용
        if syndromes.ndim == 3:
            syn = syndromes[:, -1, :]  # (N, 6)
        else:
            syn = syndromes  # (N, 6)
        
        N = syn.shape[0]
        corrections = np.zeros((N, 7), dtype=np.int8)
        
        for i in range(N):
            # X-type syndrome (앞 3개) → Z-에러 correction
            x_key = tuple(syn[i, :3].astype(int).tolist())
            z_correction = self._cache.get(x_key, np.zeros(7, dtype=np.int8))
            
            # Z-type syndrome (뒤 3개) → X-에러 correction
            z_key = tuple(syn[i, 3:].astype(int).tolist())
            x_correction = self._cache.get(z_key, np.zeros(7, dtype=np.int8))
            
            # 두 correction을 OR로 결합 (같은 큐빗이면 Y-에러)
            corrections[i] = (z_correction | x_correction).astype(np.int8)
        
        return corrections


# ==============================================================================
# Verification
# ==============================================================================
def verify_mwpm_vs_lookup():
    """
    d=3에서 MWPM Restriction Decoder와 Lookup Table이
    모든 단일 에러에 대해 동일한 결과를 내는지 검증합니다.
    """
    print("=" * 60)
    print("  MWPM Restriction Decoder Verification")
    print("=" * 60)
    
    mwpm = MWPMColorCodeDecoder()
    
    print(f"\n  PyMatching available: {HAS_PYMATCHING}")
    
    # === Test 1: 모든 단일 에러 신드롬 ===
    print(f"\n  [Test 1] All single-qubit error syndromes:")
    all_match = True
    
    for q in range(7):
        for basis_name, basis_offset in [("X-syn->Z-err", 0), ("Z-syn->X-err", 3)]:
            syn = np.zeros(6, dtype=np.float32)
            col = H_MATRIX[:, q]
            syn[basis_offset:basis_offset + 3] = col
            
            syndromes = syn.reshape(1, 1, 6)
            data_states = np.zeros((1, 7), dtype=np.int8)
            
            mwpm_corr = mwpm.decode(syndromes, data_states)
            
            # Expected: correction on qubit q
            expected = np.zeros(7, dtype=np.int8)
            expected[q] = 1
            
            match = np.array_equal(mwpm_corr[0], expected)
            status = "OK" if match else "FAIL"
            
            if not match:
                all_match = False
                print(f"  [{status}] q{q} {basis_name}: syn={col} -> MWPM={mwpm_corr[0]} (expected={expected})")
            else:
                print(f"  [{status}] q{q} {basis_name}: syn={col} -> correction on q{q}")
    
    # === Test 2: Zero syndrome ===
    print(f"\n  [Test 2] Zero syndrome:")
    syn_zero = np.zeros((1, 1, 6), dtype=np.float32)
    data_zero = np.zeros((1, 7), dtype=np.int8)
    corr_zero = mwpm.decode(syn_zero, data_zero)
    zero_ok = np.all(corr_zero == 0)
    print(f"  [{'OK' if zero_ok else 'FAIL'}] Zero syndrome -> zero correction: {zero_ok}")
    
    # === Test 3: Cache entries ===
    print(f"\n  [Test 3] Correction cache (all 3-bit syndromes):")
    for key, corr in sorted(mwpm._cache.items()):
        if sum(key) > 0:
            q = np.argmax(corr) if corr.sum() > 0 else -1
            print(f"    syn={key} -> correction on q{q} (weight={corr.sum()})")
    
    # === Test 4: Restriction consistency analysis ===
    if HAS_PYMATCHING:
        print(f"\n  [Test 4] Restriction consistency per qubit:")
        for q in range(7):
            syn = H_MATRIX[:, q].astype(np.uint8)
            consistent = []
            
            for name, (row_i, row_j) in RESTRICTIONS.items():
                rs = np.array([syn[row_i], syn[row_j]], dtype=np.uint8)
                if rs.sum() == 0:
                    continue
                
                h_r = H_MATRIX[[row_i, row_j], :]
                m_r = pymatching.Matching(csc_matrix(h_r))
                c = m_r.decode(rs)
                full_check = (H_MATRIX @ c) % 2
                
                if np.array_equal(full_check, syn):
                    consistent.append(name)
            
            method = "restriction" if consistent else "fallback"
            print(f"    q{q}: syn={list(syn)} consistent={consistent or 'none'} -> {method}")
    
    print(f"\n  Result: {'ALL PASS' if all_match and zero_ok else 'SOME FAILED'}")
    return all_match and zero_ok


if __name__ == "__main__":
    verify_mwpm_vs_lookup()