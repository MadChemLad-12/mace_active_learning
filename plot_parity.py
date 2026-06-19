import matplotlib.pyplot as plt
from ase.io import read
import numpy as np
import argparse

def create_plots(filename, prefix):
    frames = read(filename, index=':')
    
    # Extract Reference vs Predicted Forces
    # MACE outputs predictions with 'MACE_' prefix by default
    ref_f = np.concatenate([f.arrays['REF_forces'] for f in frames]).flatten()
    pred_f = np.concatenate([f.arrays['MACE_forces'] for f in frames]).flatten()

    plt.figure(figsize=(6,6))
    plt.scatter(ref_f, pred_f, alpha=0.3, s=1)
    plt.plot([ref_f.min(), ref_f.max()], [ref_f.min(), ref_f.max()], 'k--')
    plt.xlabel("Reference Forces (eV/A)")
    plt.ylabel("MACE Forces (eV/A)")
    plt.title(f"Force Parity: {prefix}")
    plt.savefig(f"{prefix}_force_parity.png")
    
    print(f"Saved parity plot to {prefix}_force_parity.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--prefix")
    args = parser.parse_args()
    create_plots(args.input, args.prefix)