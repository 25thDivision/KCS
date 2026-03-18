import numpy as np
import pymatching
from scipy.sparse import csc_matrix

class SurfaceCode2D:
    def __init__(self, distance):
        self.d = distance
        self.L = 2 * distance - 1

        # --- Topology Definition ---
        # Coordinate system: (row, col)
        # (r+c) % 2 == 0 : Data Qubit
        # (r+c) % 2 == 1 : Ancilla Qubit
        # Among them, r % 2 == 0 : Z-Stabilizer
        # Among them, r % 2 == 1 : X-Stabilizer

        self.data_qubit_coords = []
        self.z_ancilla_coords = []
        self.x_ancilla_coords = []
        self.coord_to_data_idx = {}

        for r in range(self.L):
            for c in range(self.L):
                if (r + c) % 2 == 0:
                    self.coord_to_data_idx[(r,c)] = len(self.data_qubit_coords)
                    self.data_qubit_coords.append((r,c))
                elif r % 2 == 0:
                    self.z_ancilla_coords.append((r,c))
                else:
                    self.x_ancilla_coords.append((r,c))

        self.n_data = len(self.data_qubit_coords)
        self.n_z = len(self.z_ancilla_coords)
        self.n_x = len(self.x_ancilla_coords)

        # Build Check Matrices
        self.hz = self._build_check_matrix(self.z_ancilla_coords)
        self.hx = self._build_check_matrix(self.x_ancilla_coords)

        # Initialize PyMatching Decoders
        self.matching_z = pymatching.Matching(self.hz)
        self.matching_x = pymatching.Matching(self.hx)

    def _build_check_matrix(self, ancilla_coords):
        row_indices, col_indices = [], []
        for i, (r, c) in enumerate(ancilla_coords):
            # Search neighboring data qubits
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for nr, nc in neighbors:
                if (nr, nc) in self.coord_to_data_idx:
                    row_indices.append(i)
                    col_indices.append(self.coord_to_data_idx[(nr, nc)])

        # shape=(ancilla_count, data_qubit_count)
        return csc_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                          shape=(len(ancilla_coords), self.n_data))

    def generate_batch(self, p_data, p_measure, n_samples):
        # Error Injection
        x_noise = (np.random.random((n_samples, self.n_data)) < p_data).astype(np.uint8)
        z_noise = (np.random.random((n_samples, self.n_data)) < p_data).astype(np.uint8)

        # Syndrome Extraction
        # Hz checks X errors, Hx checks Z errors
        z_syndrome = (x_noise @ self.hz.T) % 2
        x_syndrome = (z_noise @ self.hx.T) % 2

        # Measurement Error
        if p_measure > 0:
            z_syndrome = (z_syndrome + (np.random.random(z_syndrome.shape) < p_measure)) % 2
            x_syndrome = (x_syndrome + (np.random.random(x_syndrome.shape) < p_measure)) % 2

        # Label Correction (MWPM)
        refined_label_x = np.zeros_like(x_noise)
        refined_label_z = np.zeros_like(z_noise)

        for i in range(n_samples):
            # decode returns the most likely error pattern (fewest errors)
            refined_label_x[i] = self.matching_z.decode(z_syndrome[i])
            refined_label_z[i] = self.matching_x.decode(x_syndrome[i])

        # Map back to Image Grid (N, 2, L, L)
        X_img = np.zeros((n_samples, 2, self.L, self.L), dtype=np.float32)
        Y_img = np.zeros((n_samples, 2, self.L, self.L), dtype=np.float32)

        # Fill Syndromes (Input)
        for idx, (r, c) in enumerate(self.z_ancilla_coords):
            X_img[:, 0, r, c] = z_syndrome[:, idx]
        for idx, (r, c) in enumerate(self.x_ancilla_coords):
            X_img[:, 1, r, c] = x_syndrome[:, idx]

        # Fill Errors (Label)
        for idx, (r, c) in enumerate(self.data_qubit_coords):
            Y_img[:, 0, r, c] = refined_label_x[:, idx]
            Y_img[:, 1, r, c] = refined_label_z[:, idx]

        return X_img, Y_img