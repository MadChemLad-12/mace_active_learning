"""
CP2K NEB Output Parser + MACE Comparison
=========================================
Parses CP2K IT-NEB/CI-NEB output and compares the energy profile
and per-image forces against a MACE MLP model.

Usage:
    python parse_neb_compare_mace.py \
        --cp2k_out Close0.75PtO2neb.out \
        --replica_dir . \
        --mace_model /path/to/mace_model.model \
        --n_replicas 10

Output:
    - neb_comparison.png  : energy profile + force RMSE per image
    - neb_results.csv     : per-image energies and metrics
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ── ASE / MACE (imported lazily so the parser still works without them) ──
try:
    from ase.io import read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

try:
    from mace.calculators import MACECalculator
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False

# ─────────────────────────────────────────────
# 1. CP2K OUTPUT PARSER
# ─────────────────────────────────────────────
from ase.units import Hartree
# Hartree → eV conversion
HA2EV = Hartree

def parse_cp2k_neb_output(outfile: str) -> dict:
    """
    Parse a CP2K BAND (NEB) output file.

    Returns a dict with:
        'iterations'   : list of dicts, one per BAND step
        'final_energies': np.array of final per-replica energies in eV
        'final_forces' : list of np.arrays (n_atoms, 3) in eV/Å, one per replica
        'reaction_coord': np.array of cumulative path lengths (Å)
    """
    outfile = Path(outfile)
    text = outfile.read_text()

    # ── Per-step energy table ──
    # CP2K prints something like:
    #  BAND TOTAL ENERGY [au]:     -1234.5  -1234.4  ...
    step_energies = []
    for line in text.splitlines():
        if "BAND TOTAL ENERGY" in line:
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if nums:
                step_energies.append([float(x) * HA2EV for x in nums])

    # ── Final replica energies (last occurrence) ──
    final_energies = np.array(step_energies[-1]) if step_energies else np.array([])

    # Shift so minimum = 0 (barrier relative to lowest image)
    if final_energies.size:
        final_energies -= final_energies.min()

    # ── Per-replica max forces (from BAND output) ──
    # CP2K prints: "          1    -1234.5   0.00312   0.00198" etc.
    # for each replica in the convergence table
    max_forces = []
    conv_block = re.findall(
        r"BAND STEP.*?(?=BAND STEP|\Z)", text, re.DOTALL
    )
    if conv_block:
        last_block = conv_block[-1]
        for line in last_block.splitlines():
            m = re.match(r"\s+(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                max_forces.append(float(m.group(3)))  # MAX_FORCE column

    return {
        "step_energies": step_energies,          # all iterations
        "final_energies": final_energies,         # eV, zero-shifted
        "max_forces_ha_bohr": max_forces,         # Ha/Bohr per replica
        "n_steps": len(step_energies),
    }


def parse_replica_xyz_files(replica_dir: str, n_replicas: int,
                            prefix: str = "", suffix: str = ".xyz") -> list:
    """
    Read per-replica XYZ files produced by CP2K NEB.

    CP2K writes:  <PROJECT>-pos-Replica_nr_X-1.xyz
    or your input files:  00.xyz, 01.xyz, ...

    Returns list of ASE Atoms objects.
    """
    if not ASE_AVAILABLE:
        raise RuntimeError("ASE not installed; cannot read XYZ files.")

    replica_dir = Path(replica_dir)
    atoms_list = []

    # Try CP2K output naming first, then fallback to input naming
    for i in range(n_replicas):
        candidates = [
            # CP2K NEB output files
            replica_dir / f"{prefix}-pos-Replica_nr_{i+1}-1.xyz",
            replica_dir / f"Replica_nr_{i+1}.xyz",
            # Your input file naming (00.xyz … 09.xyz)
            replica_dir / f"{i:02d}.xyz",
            replica_dir / f"{i}.xyz",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Cannot find XYZ for replica {i}. Tried: {candidates}"
            )
        # Read last frame (NEB may write trajectory per replica)
        frames = read(str(found), index=":")
        atoms_list.append(frames[-1])

    return atoms_list


def compute_reaction_coordinate(atoms_list: list) -> np.ndarray:
    """Cumulative Cartesian displacement along the NEB path (Å)."""
    coords = np.array([a.get_positions() for a in atoms_list])
    diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=(1, 2))
    return np.concatenate([[0.0], np.cumsum(diffs)])


# ─────────────────────────────────────────────
# 2. MACE EVALUATION
# ─────────────────────────────────────────────

def evaluate_mace(atoms_list: list, model_path: str,
                  device: str = "cpu") -> dict:
    """
    Run MACE single-point calculations on each replica.

    Returns:
        'energies'     : np.array of energies in eV (zero-shifted)
        'force_arrays' : list of (n_atoms, 3) force arrays in eV/Å
        'max_forces'   : np.array of max force magnitude per image
    """
    if not MACE_AVAILABLE:
        raise RuntimeError("MACE not installed.")

    calc = MACECalculator(model_paths=model_path, device=device,
                          default_dtype="float64")

    energies, force_arrays, max_forces = [], [], []
    for atoms in atoms_list:
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        energies.append(e)
        force_arrays.append(f)
        max_forces.append(np.max(np.linalg.norm(f, axis=1)))

    energies = np.array(energies)
    energies -= energies.min()   # zero-shift

    return {
        "energies": energies,
        "force_arrays": force_arrays,
        "max_forces": np.array(max_forces),
    }


# ─────────────────────────────────────────────
# 3. COMPARISON METRICS
# ─────────────────────────────────────────────

def force_rmse_per_image(cp2k_forces: list, mace_forces: list) -> np.ndarray:
    """Per-image force RMSE between CP2K and MACE (eV/Å)."""
    rmse = []
    for f_cp2k, f_mace in zip(cp2k_forces, mace_forces):
        diff = f_cp2k - f_mace
        rmse.append(np.sqrt(np.mean(diff**2)))
    return np.array(rmse)


def energy_mae(cp2k_e: np.ndarray, mace_e: np.ndarray) -> float:
    return np.mean(np.abs(cp2k_e - mace_e))


def barrier_error(cp2k_e: np.ndarray, mace_e: np.ndarray) -> tuple:
    """Forward and reverse barrier error (meV)."""
    cp2k_fwd = cp2k_e.max() - cp2k_e[0]
    mace_fwd  = mace_e.max() - mace_e[0]
    cp2k_rev  = cp2k_e.max() - cp2k_e[-1]
    mace_rev  = mace_e.max() - mace_e[-1]
    return (abs(cp2k_fwd - mace_fwd) * 1000,   # meV
            abs(cp2k_rev - mace_rev) * 1000)


# ─────────────────────────────────────────────
# 4. PLOTTING
# ─────────────────────────────────────────────

def plot_comparison(rxn_coord, cp2k_energies, mace_energies=None,
                    force_rmse=None, outfile="neb_comparison.png"):
    """
    Two-panel figure:
      Top    : energy profile CP2K vs MACE
      Bottom : per-image force RMSE (if forces available)
    """
    n_panels = 2 if force_rmse is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    # ── Energy profile ──
    ax = axes[0]
    ax.plot(rxn_coord, cp2k_energies * 1000, "o-", color="#2166ac",
            lw=2, ms=7, label="CP2K DFT (PBE-D3)")
    if mace_energies is not None:
        ax.plot(rxn_coord, mace_energies * 1000, "s--", color="#d6604d",
                lw=2, ms=7, label="MACE MLP")
    ax.set_ylabel("Relative energy (meV)", fontsize=12)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.legend(fontsize=11)
    ax.set_title("NEB Energy Profile: CP2K vs MACE", fontsize=13)

    # ── Force RMSE ──
    if force_rmse is not None:
        ax2 = axes[1]
        ax2.bar(rxn_coord, force_rmse * 1000, width=rxn_coord[1] * 0.6,
                color="#4dac26", alpha=0.8, label="Force RMSE")
        ax2.set_ylabel("Force RMSE (meV/Å)", fontsize=12)
        ax2.set_xlabel("Reaction coordinate (Å)", fontsize=12)
        ax2.legend(fontsize=11)
        ax2.set_title("Per-image Force RMSE (CP2K − MACE)", fontsize=13)
    else:
        axes[-1].set_xlabel("Reaction coordinate (Å)", fontsize=12)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outfile}")
    return fig


# ─────────────────────────────────────────────
# 5. CONVERGENCE PLOT (bonus)
# ─────────────────────────────────────────────

def plot_convergence(step_energies: list, outfile="neb_convergence.png"):
    """Plot max and min image energy vs NEB iteration."""
    arr = np.array(step_energies)   # shape (n_steps, n_replicas)
    barriers = arr.max(axis=1) - arr.min(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(barriers * 1000, "k-o", ms=5)
    ax.set_xlabel("NEB iteration", fontsize=12)
    ax.set_ylabel("Barrier estimate (meV)", fontsize=12)
    ax.set_title("NEB Convergence: Barrier vs Iteration", fontsize=13)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outfile}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse CP2K NEB and compare to MACE")
    parser.add_argument("--cp2k_out",    required=True,  help="CP2K .out file")
    parser.add_argument("--replica_dir", default=".",    help="Directory with replica XYZs")
    parser.add_argument("--project",     default="",     help="CP2K PROJECT_NAME (for output file naming)")
    parser.add_argument("--mace_model",  default=None,   help="Path to MACE .model file")
    parser.add_argument("--n_replicas",  type=int, default=10)
    parser.add_argument("--device",      default="cpu",  help="cpu or cuda")
    parser.add_argument("--out_dir",     default=".",    help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("── Parsing CP2K output ──")
    cp2k = parse_cp2k_neb_output(args.cp2k_out)
    print(f"  NEB steps completed : {cp2k['n_steps']}")
    print(f"  Final energies (eV) : {np.round(cp2k['final_energies'], 4)}")
    if cp2k['final_energies'].size:
        print(f"  Forward barrier     : {cp2k['final_energies'].max()*1000:.1f} meV")

    # ── Plot convergence ──
    if cp2k['step_energies']:
        plot_convergence(cp2k['step_energies'],
                         outfile=str(out_dir / "neb_convergence.png"))

    # ── Read replica geometries ──
    print("\n── Reading replica XYZ files ──")
    atoms_list = parse_replica_xyz_files(
        args.replica_dir, args.n_replicas, prefix=args.project
    )
    rxn_coord = compute_reaction_coordinate(atoms_list)
    print(f"  Path length: {rxn_coord[-1]:.3f} Å")

    # ── CP2K forces from XYZ comment lines (if present) ──
    # CP2K writes forces to separate files; skip if unavailable
    cp2k_forces = None

    # ── MACE evaluation ──
    mace_results = None
    force_rmse = None

    if args.mace_model:
        print("\n── Evaluating MACE model ──")
        mace_results = evaluate_mace(atoms_list, args.mace_model, args.device)
        print(f"  MACE energies (eV): {np.round(mace_results['energies'], 4)}")
        print(f"  MACE forward barrier: {mace_results['energies'].max()*1000:.1f} meV")

        # ── Metrics ──
        mae = energy_mae(cp2k['final_energies'], mace_results['energies'])
        fwd_err, rev_err = barrier_error(cp2k['final_energies'],
                                         mace_results['energies'])
        print(f"\n── Comparison Metrics ──")
        print(f"  Energy MAE          : {mae*1000:.2f} meV")
        print(f"  Forward barrier Δ   : {fwd_err:.1f} meV")
        print(f"  Reverse barrier Δ   : {rev_err:.1f} meV")

    # ── Save CSV ──
    df_data = {"replica": list(range(args.n_replicas)),
               "rxn_coord_A": rxn_coord}
    if cp2k['final_energies'].size == args.n_replicas:
        df_data["cp2k_energy_eV"] = cp2k['final_energies']
        df_data["cp2k_energy_meV"] = cp2k['final_energies'] * 1000
    if mace_results:
        df_data["mace_energy_eV"]  = mace_results['energies']
        df_data["mace_energy_meV"] = mace_results['energies'] * 1000
        df_data["mace_max_force_eV_A"] = mace_results['max_forces']
        if cp2k['final_energies'].size == args.n_replicas:
            df_data["energy_diff_meV"] = (
                (cp2k['final_energies'] - mace_results['energies']) * 1000
            )

    df = pd.DataFrame(df_data)
    csv_path = out_dir / "neb_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(df.to_string(index=False))

    # ── Plot ──
    print("\n── Generating plot ──")
    plot_comparison(
        rxn_coord,
        cp2k['final_energies'] if cp2k['final_energies'].size else np.zeros(args.n_replicas),
        mace_energies=mace_results['energies'] if mace_results else None,
        force_rmse=force_rmse,
        outfile=str(out_dir / "neb_comparison.png"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()