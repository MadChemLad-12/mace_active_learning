import torch
from mace.calculators import MACECalculator
from ase.io import read
import numpy as np
import os
from configs.constants import CLEAN_TRAIN, FOUNDATION_MODEL_PATH
from configs.round_configs.schema import ActivePipelineConfig
from helpers.imp_helper import imp_highest_config
active_pipeline = imp_highest_config("active_pipeline")

MODEL_PATH = FOUNDATION_MODEL_PATH
POOL_FILE  = CLEAN_TRAIN

# Load pool and sort by volume largest first
pool = read(POOL_FILE, index=":")
pool_sorted = sorted(
    pool, 
    key=len, 
    reverse=True
)
config = active_pipeline.ActivePipelineConfig()
# config = ActivePipelineConfig()
calc = MACECalculator(model_paths=MODEL_PATH, device=config.device, default_dtype=config.dtype)

print(f"{'Volume (Å³)':<15} {'N atoms':<10} {'Index' :<10} {'Status':<10} ")
print("-" * 40)

for idx, atoms in enumerate(pool_sorted):
    vol = atoms.get_volume()
    n_atoms = len(atoms)
    index = idx
    
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    try:
        torch.cuda.reset_peak_memory_stats()
        atoms_copy.get_potential_energy()
        mem = torch.cuda.max_memory_allocated() / 1e9  # GB
        print(f"{vol:<15.1f} {n_atoms:<10} {index:<10} OK  ({mem:.2f} GB)")
    except torch.cuda.OutOfMemoryError:
        print(f"{vol:<15.1f} {n_atoms:<10} {index:<10} OOM ← limit is here")
        torch.cuda.empty_cache()
        break