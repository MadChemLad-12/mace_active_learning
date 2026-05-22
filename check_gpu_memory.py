import torch
from mace.calculators import MACECalculator
from ase.io import read
import numpy as np

MODEL_PATH = "mace-mp-0b3-medium-float32.model"
POOL_FILE  = "master_train_pool.xyz"

# Load pool and sort by volume largest first
pool = read(POOL_FILE, index=":")
pool_sorted = sorted(
    pool, 
    key=lambda x: (x.get_volume(), len(x)), 
    reverse=True
)
calc = MACECalculator(model_paths=MODEL_PATH, device="cuda", default_dtype="float32")

print(f"{'Volume (Å³)':<15} {'N atoms':<10} {'Status':<10}")
print("-" * 40)

for atoms in pool_sorted:
    vol = atoms.get_volume()
    n_atoms = len(atoms)
    
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    try:
        torch.cuda.reset_peak_memory_stats()
        atoms_copy.get_potential_energy()
        mem = torch.cuda.max_memory_allocated() / 1e9  # GB
        print(f"{vol:<15.1f} {n_atoms:<10} OK  ({mem:.2f} GB)")
    except torch.cuda.OutOfMemoryError:
        print(f"{vol:<15.1f} {n_atoms:<10} OOM ← limit is here")
        torch.cuda.empty_cache()
        break