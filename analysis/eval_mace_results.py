#!/usr/bin/env python
"""
Build a per-config_type RMSE error table and parity plots (E, F) from an
extxyz file produced by `mace_eval_configs`.

Usage:
    python eval_mace_results.py --xyz eval_output.xyz --out_dir ./eval_plots

Assumes standard mace_eval_configs output keys:
    atoms.info['REF_energy']          - DFT energy
    atoms.info['MACE_energy']         - predicted energy
    atoms.arrays['REF_forces']        - DFT forces (N,3)
    atoms.arrays['MACE_forces']       - predicted forces (N,3)
    atoms.info['config_type']         - system label (e.g. Pt_Nafion, Naf-Naf, ...)

If your keys differ (e.g. no 'MACE_' prefix, or 'energy'/'REF_energy' swapped),
adjust the ENERGY_REF_KEY / ENERGY_PRED_KEY / FORCES_REF_KEY / FORCES_PRED_KEY
constants below.
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from ase.io import read

# ---- adjust if your file uses different keys ----
ENERGY_REF_KEY = "REF_energy"
ENERGY_PRED_KEY = "MACE_energy"
FORCES_REF_KEY = "REF_forces"
FORCES_PRED_KEY = "MACE_forces"
CONFIG_TYPE_KEY = "system_type"
# ---------------------------------------------------


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_data(xyz_path):
    frames = read(xyz_path, index=":")
    print(f"Loaded {len(frames)} frames from {xyz_path}")

    data = defaultdict(lambda: {
        "e_ref": [], "e_pred": [], "n_atoms": [],
        "f_ref": [], "f_pred": [],
    })

    missing_energy = 0
    missing_forces = 0

    for atoms in frames:
        ctype = atoms.info.get(CONFIG_TYPE_KEY, "unknown")

        if ENERGY_REF_KEY in atoms.info and ENERGY_PRED_KEY in atoms.info:
            data[ctype]["e_ref"].append(atoms.info[ENERGY_REF_KEY])
            data[ctype]["e_pred"].append(atoms.info[ENERGY_PRED_KEY])
            data[ctype]["n_atoms"].append(len(atoms))
        else:
            missing_energy += 1

        if FORCES_REF_KEY in atoms.arrays and FORCES_PRED_KEY in atoms.arrays:
            data[ctype]["f_ref"].append(atoms.arrays[FORCES_REF_KEY])
            data[ctype]["f_pred"].append(atoms.arrays[FORCES_PRED_KEY])
        else:
            missing_forces += 1

    if missing_energy or missing_forces:
        print(f"WARNING: {missing_energy} frames missing energy keys, "
              f"{missing_forces} frames missing force keys. "
              f"Check ENERGY_*/FORCES_* key names at top of script.")

    return data


def build_table(data):
    rows = []
    for ctype, d in sorted(data.items()):
        if not d["e_ref"]:
            continue
        e_ref = np.array(d["e_ref"])
        e_pred = np.array(d["e_pred"])
        n_atoms = np.array(d["n_atoms"])

        e_rmse_per_atom = rmse(e_ref / n_atoms, e_pred / n_atoms) * 1000  # meV/atom

        if d["f_ref"]:
            f_ref = np.concatenate([f.flatten() for f in d["f_ref"]])
            f_pred = np.concatenate([f.flatten() for f in d["f_pred"]])
            f_rmse = rmse(f_ref, f_pred) * 1000  # meV/A
            f_rel = 100 * np.sqrt(np.mean((f_ref - f_pred) ** 2)) / np.sqrt(np.mean(f_ref ** 2))
        else:
            f_rmse, f_rel = float("nan"), float("nan")

        rows.append((ctype, len(d["e_ref"]), e_rmse_per_atom, f_rmse, f_rel))
    return rows


def print_table(rows):
    header = f"{'config_type':<20} {'n':>5} {'RMSE E / meV/atom':>18} {'RMSE F / meV/A':>16} {'rel F RMSE %':>13}"
    print(header)
    print("-" * len(header))
    for ctype, n, e_rmse, f_rmse, f_rel in rows:
        print(f"{ctype:<20} {n:>5} {e_rmse:>18.2f} {f_rmse:>16.2f} {f_rel:>13.2f}")


def write_table_csv(rows, out_path):
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_type", "n_frames", "RMSE_E_meV_per_atom", "RMSE_F_meV_per_A", "rel_F_RMSE_pct"])
        for row in rows:
            w.writerow(row)
    print(f"Table written to {out_path}")


def make_parity_plots(data, out_dir):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # combined energy parity (colored by config_type)
    fig, ax = plt.subplots(figsize=(6, 6))
    all_e_ref, all_e_pred = [], []
    for ctype, d in sorted(data.items()):
        if not d["e_ref"]:
            continue
        n_atoms = np.array(d["n_atoms"])
        e_ref = np.array(d["e_ref"]) / n_atoms * 1000
        e_pred = np.array(d["e_pred"]) / n_atoms * 1000
        ax.scatter(e_ref, e_pred, s=14, alpha=0.6, label=ctype)
        all_e_ref.extend(e_ref)
        all_e_pred.extend(e_pred)
    lims = [min(all_e_ref + all_e_pred), max(all_e_ref + all_e_pred)]
    ax.plot(lims, lims, "k--", lw=1, zorder=0)
    ax.set_xlabel("DFT energy (meV/atom)")
    ax.set_ylabel("MACE energy (meV/atom)")
    ax.set_title("Energy parity by config_type")
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_parity.png", dpi=200)
    plt.close(fig)

    # combined force parity
    fig, ax = plt.subplots(figsize=(6, 6))
    all_f_ref, all_f_pred = [], []
    for ctype, d in sorted(data.items()):
        if not d["f_ref"]:
            continue
        f_ref = np.concatenate([f.flatten() for f in d["f_ref"]])
        f_pred = np.concatenate([f.flatten() for f in d["f_pred"]])
        # subsample for plotting if huge
        if len(f_ref) > 20000:
            idx = np.random.choice(len(f_ref), 20000, replace=False)
            f_ref_plot, f_pred_plot = f_ref[idx], f_pred[idx]
        else:
            f_ref_plot, f_pred_plot = f_ref, f_pred
        ax.scatter(f_ref_plot, f_pred_plot, s=4, alpha=0.3, label=ctype)
        all_f_ref.extend(f_ref)
        all_f_pred.extend(f_pred)
    lims = [min(all_f_ref + all_f_pred), max(all_f_ref + all_f_pred)]
    ax.plot(lims, lims, "k--", lw=1, zorder=0)
    ax.set_xlabel("DFT force component (eV/A)")
    ax.set_ylabel("MACE force component (eV/A)")
    ax.set_title("Force parity by config_type")
    ax.legend(fontsize=7, markerscale=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "force_parity.png", dpi=200)
    plt.close(fig)

    # per-config_type small multiples for energy (useful when scales differ a lot)
    ctypes = [c for c, d in data.items() if d["e_ref"]]
    n = len(ctypes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for i, ctype in enumerate(sorted(ctypes)):
        ax = axes[i // ncols][i % ncols]
        d = data[ctype]
        n_atoms = np.array(d["n_atoms"])
        e_ref = np.array(d["e_ref"]) / n_atoms * 1000
        e_pred = np.array(d["e_pred"]) / n_atoms * 1000
        ax.scatter(e_ref, e_pred, s=14, alpha=0.6)
        lims = [min(e_ref.min(), e_pred.min()), max(e_ref.max(), e_pred.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_title(f"{ctype}\nRMSE={rmse(e_ref, e_pred):.2f} meV/atom", fontsize=9)
        ax.set_xlabel("DFT (meV/atom)", fontsize=8)
        ax.set_ylabel("MACE (meV/atom)", fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_parity_by_type.png", dpi=200)
    plt.close(fig)

    print(f"Plots written to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True, help="Path to mace_eval_configs output extxyz")
    ap.add_argument("--out_dir", default="./eval_output", help="Directory for plots + csv table")
    args = ap.parse_args()

    data = load_data(args.xyz)
    rows = build_table(data)
    print_table(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table_csv(rows, out_dir / "error_table.csv")
    make_parity_plots(data, out_dir)


if __name__ == "__main__":
    main()