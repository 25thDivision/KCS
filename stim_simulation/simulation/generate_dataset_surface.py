import numpy as np
import os
import time
from tqdm import tqdm
from generators.surface_code import SurfaceCode2D

# --- Settings ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../dataset/surface_code")
os.makedirs(OUTPUT_DIR, exist_ok=True)

d_list = [3, 5, 7]
p_list = [0.005, 0.01, 0.05]

TARGET_TRAIN = 1000000
CHUNK_SIZE   = 10000
NUM_TEST     = 2000

print(f"=== Generating Dataset (Target: {TARGET_TRAIN}) ===")
print(f"Save Directory: {OUTPUT_DIR}\n")

for d in d_list:
    code = SurfaceCode2D(d) # Instantiate class
    
    for p in p_list:
        start_time = time.time()
        train_file = f"{OUTPUT_DIR}/surface_train_d{d}_p{p}.npz"
        test_file = f"{OUTPUT_DIR}/surface_test_d{d}_p{p}.npz"
        
        # Skip if files already exist
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"✓ [Skip] Already exists: d={d}, p={p}")
            continue
            
        print(f"processing d={d}, p={p}...")
        
        # Train data generation (Chunking)
        X_chunks = []
        Y_chunks = []
            
        # Repeat 1 million samples in chunks of 10,000
        num_chunks = TARGET_TRAIN // CHUNK_SIZE
        
        for i in tqdm(range(num_chunks), desc=f"  Generating {TARGET_TRAIN} samples"):
            # Reuse generate_batch function
            x_part, y_part = code.generate_batch(p, p, CHUNK_SIZE)
            X_chunks.append(x_part)
            Y_chunks.append(y_part)
            
        # Concatenate all chunks
        X_train = np.concatenate(X_chunks, axis=0)
        Y_train = np.concatenate(Y_chunks, axis=0)
        
        # Save Train dataset
        np.savez_compressed(train_file, X=X_train, y=Y_train)
        
        # Test data generation
        X_test, Y_test = code.generate_batch(p, p, NUM_TEST)
        np.savez_compressed(test_file, X=X_test, y=Y_test)
        
        elapsed = (time.time() - start_time) / 60
        print(f">>> Save completed ({elapsed:.1f} minutes elapsed)")

print("\n >>> All data generation completed!")