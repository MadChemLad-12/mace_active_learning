#!/usr/bin/env python
"""
Unified MACE evaluation tool:
Calculates aggregate RMSE tables (Energy & Force), per-frame Force RMSE distributions,
and generates parity/distribution plots split by config_type from an extxyz file.

Usage:
    python evaluate_mace.py --xyz eval_output.xyz --out_dir ./eval_results
    python evaluate_mace.py --xyz eval_output.xyz --out_dir ./eval_results --types Pt_Nafion Pt_water
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

# ---- Key Configuration (Adjust to match your extxyz attributes) ----
ENERGY_REF_KEY = "REF_energy"
ENERGY_PRED_KEY = "MACE_energy"
FORCES_REF_KEY = "REF_forces"
FORCES_PRED_KEY = "MACE_forces"
CONFIG_TYPE_KEY = "system_type"  # Change to "system_type" if needed
# ----------------------------------------------------------------------


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_all_data(xyz_path):
    """Single pass file reader populating structure and metric arrays."""
    frames = read(xyz_path, index=":")
    print(f"Loaded {len(frames)} frames from {xyz_path}")

    data = defaultdict(
        lambda: {
            "e_ref": [],
            "e_pred": [],
            "n_atoms": [],
            "f_ref": [],
            "f_pred": [],
            "frame_force_rmses": [],
            "frame_rel_force_rmses": [],
        }
    )

    missing_energy = 0
    missing_forces = 0

    for atoms in frames:
        ctype = atoms.info.get(CONFIG_TYPE_KEY, "unknown")

        # Process Energy
        has_e = ENERGY_REF_KEY in atoms.info and ENERGY_PRED_KEY in atoms.info
        if has_e:
            data[ctype]["e_ref"].append(atoms.info[ENERGY_REF_KEY])
            data[ctype]["e_pred"].append(atoms.info[ENERGY_PRED_KEY])
            data[ctype]["n_atoms"].append(len(atoms))
        else:
            missing_energy += 1

        # Process Forces
        has_f = (
            FORCES_REF_KEY in atoms.arrays and FORCES_PRED_KEY in atoms.arrays
        )
        if has_f:
            f_ref = atoms.arrays[FORCES_REF_KEY]
            f_pred = atoms.arrays[FORCES_PRED_KEY]

            data[ctype]["f_ref"].append(f_ref)
            data[ctype]["f_pred"].append(f_pred)

            # Per-frame force RMSE calculation (meV/A)
            frame_rmse = float(np.sqrt(np.mean((f_ref - f_pred) ** 2))) * 1000
            data[ctype]["frame_force_rmses"].append(frame_rmse)

            # Per-frame relative force RMSE calculation (%)
            ref_sq_mean = np.mean(f_ref**2)
            if ref_sq_mean > 1e-12:
                frame_rel_rmse = 100 * np.sqrt(np.mean((f_ref - f_pred) ** 2)) / np.sqrt(ref_sq_mean)
            else:
                frame_rel_rmse = float("nan")
            data[ctype]["frame_rel_force_rmses"].append(frame_rel_rmse)
        else:
            missing_forces += 1

    if missing_energy or missing_forces:
        print(
            f"WARNING: {missing_energy} frames missing energy keys, "
            f"{missing_forces} frames missing force keys."
        )

    return data


def build_and_write_summary_tables(data, out_dir):
    """Print and write aggregate error metrics and distribution shape summaries."""
    summary_rows = []
    dist_rows = []

    print(
        f"\n{'config_type':<20} {'n':>5} {'RMSE E (meV/at)':>18} {'RMSE F (meV/A)':>16} {'rel F RMSE %':>13}"
    )
    print("-" * 75)

    for ctype, d in sorted(data.items()):
        if not d["e_ref"] and not d["f_ref"]:
            continue

        n_frames = len(d["e_ref"]) if d["e_ref"] else len(d["f_ref"])

        # Aggregate metrics
        if d["e_ref"]:
            e_ref = np.array(d["e_ref"])
            e_pred = np.array(d["e_pred"])
            n_atoms = np.array(d["n_atoms"])
            e_rmse = rmse(e_ref / n_atoms, e_pred / n_atoms) * 1000
        else:
            e_rmse = float("nan")

        if d["f_ref"]:
            f_ref = np.concatenate([f.flatten() for f in d["f_ref"]])
            f_pred = np.concatenate([f.flatten() for f in d["f_pred"]])
            f_rmse = rmse(f_ref, f_pred) * 1000
            f_rel = (
                100
                * np.sqrt(np.mean((f_ref - f_pred) ** 2))
                / np.sqrt(np.mean(f_ref**2))
            )
        else:
            f_rmse, f_rel = float("nan"), float("nan")

        print(
            f"{ctype:<20} {n_frames:>5} {e_rmse:>18.2f} {f_rmse:>16.2f} {f_rel:>13.2f}"
        )
        summary_rows.append((ctype, n_frames, e_rmse, f_rmse, f_rel))

        # Force Distribution Stats
        if d["frame_force_rmses"]:
            v = np.array(d["frame_force_rmses"])
            mean, median = v.mean(), np.median(v)
            p90, p95, vmax = (
                np.percentile(v, 90),
                np.percentile(v, 95),
                v.max(),
            )
            
            # Aggregate relative force RMSE across frames for this config_type
            rel_v = np.array(d["frame_rel_force_rmses"])
            mean_rel_f = float(np.nanmean(rel_v)) if len(rel_v) > 0 else float("nan")

            tail_ratio = (vmax - p95) / (p95 - median + 1e-9)
            shape = (
                "long tail (few outliers)" if tail_ratio > 2 else "broad/smooth"
            )
            # relative force RMSE inserted directly after 'n'
            dist_rows.append(
                (ctype, len(v), mean_rel_f, mean, median, p90, p95, vmax, shape)
            )

    # Save CSVs
    with open(out_dir / "error_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config_type",
                "n_frames",
                "RMSE_E_meV_per_atom",
                "RMSE_F_meV_per_A",
                "rel_F_RMSE_pct",
            ]
        )
        w.writerows(summary_rows)

    with open(out_dir / "force_rmse_distribution_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config_type",
                "n",
                "rel_F_RMSE_pct",
                "mean_meV_A",
                "median_meV_A",
                "p90_meV_A",
                "p95_meV_A",
                "max_meV_A",
                "shape_guess",
            ]
        )
        w.writerows(dist_rows)

    print(f"\nTables saved to {out_dir}/")


def plot_parity(data, out_dir):
    """Generate global parity plots and per-type grid parity plots."""
    # 1. Combined Energy Parity
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

    if all_e_ref:
        lims = [min(all_e_ref + all_e_pred), max(all_e_ref + all_e_pred)]
        ax.plot(lims, lims, "k--", lw=1, zorder=0)
        ax.set_xlabel("DFT energy (meV/atom)")
        ax.set_ylabel("MACE energy (meV/atom)")
        ax.set_title("Energy parity by config_type")
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "energy_parity.png", dpi=200)
    plt.close(fig)

    # 2. Combined Force Parity
    fig, ax = plt.subplots(figsize=(6, 6))
    all_f_ref, all_f_pred = [], []
    for ctype, d in sorted(data.items()):
        if not d["f_ref"]:
            continue
        f_ref = np.concatenate([f.flatten() for f in d["f_ref"]])
        f_pred = np.concatenate([f.flatten() for f in d["f_pred"]])

        if len(f_ref) > 20000:
            idx = np.random.choice(len(f_ref), 20000, replace=False)
            f_ref_plot, f_pred_plot = f_ref[idx], f_pred[idx]
        else:
            f_ref_plot, f_pred_plot = f_ref, f_pred

        ax.scatter(f_ref_plot, f_pred_plot, s=4, alpha=0.3, label=ctype)
        all_f_ref.extend(f_ref)
        all_f_pred.extend(f_pred)

    if all_f_ref:
        lims = [min(all_f_ref + all_f_pred), max(all_f_ref + all_f_pred)]
        ax.plot(lims, lims, "k--", lw=1, zorder=0)
        ax.set_xlabel("DFT force component (eV/A)")
        ax.set_ylabel("MACE force component (eV/A)")
        ax.set_title("Force parity by config_type")
        ax.legend(fontsize=7, markerscale=3, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "force_parity.png", dpi=200)
    plt.close(fig)

    # 3. Small Multiples Energy Parity
    ctypes = [c for c, d in data.items() if d["e_ref"]]
    if ctypes:
        ncols = 3
        nrows = int(np.ceil(len(ctypes) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
        )

        for i, ctype in enumerate(sorted(ctypes)):
            ax = axes[i // ncols][i % ncols]
            d = data[ctype]
            n_atoms = np.array(d["n_atoms"])
            e_ref = np.array(d["e_ref"]) / n_atoms * 1000
            e_pred = np.array(d["e_pred"]) / n_atoms * 1000
            ax.scatter(e_ref, e_pred, s=14, alpha=0.6)
            lims = [
                min(e_ref.min(), e_pred.min()),
                max(e_ref.max(), e_pred.max()),
            ]
            ax.plot(lims, lims, "k--", lw=1)
            ax.set_title(
                f"{ctype}\nRMSE={rmse(e_ref, e_pred):.2f} meV/atom", fontsize=9
            )
            ax.set_xlabel("DFT (meV/atom)", fontsize=8)
            ax.set_ylabel("MACE (meV/atom)", fontsize=8)

        for j in range(len(ctypes), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout(h_pad=1.5, w_pad=1.5)
        fig.savefig(out_dir / "energy_parity_by_type.png", dpi=200)
        plt.close(fig)


def plot_distributions(data, out_dir, types_filter=None):
    """Generate histograms and boxplots for per-frame force RMSE distributions."""
    ctypes = [
        c
        for c in data
        if data[c]["frame_force_rmses"]
        and (types_filter is None or c in types_filter)
    ]
    if not ctypes:
        return

    ctypes = sorted(
        ctypes, key=lambda c: -np.mean(data[c]["frame_force_rmses"])
    )

    # Histograms
    ncols = 3
    nrows = int(np.ceil(len(ctypes) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False
    )

    for i, ctype in enumerate(ctypes):
        ax = axes[i // ncols][i % ncols]
        v = np.array(data[ctype]["frame_force_rmses"])
        ax.hist(
            v,
            bins=min(30, max(5, len(v) // 2)),
            color="steelblue",
            edgecolor="white",
        )
        median, p95 = np.median(v), np.percentile(v, 95)
        ax.axvline(
            median,
            color="k",
            linestyle="--",
            lw=1,
            label=f"median={median:.0f}",
        )
        ax.axvline(
            p95, color="firebrick", linestyle="--", lw=1, label=f"p95={p95:.0f}"
        )
        ax.set_title(f"{ctype} (n={len(v)})", fontsize=9)
        ax.set_xlabel("per-frame force RMSE (meV/A)", fontsize=8)
        ax.legend(fontsize=7)

    for j in range(len(ctypes), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        "Per-frame force RMSE distributions by config_type", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "force_rmse_distributions.png", dpi=200)
    plt.close(fig)

    # Boxplot
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(ctypes)), 5))
    plot_data = [data[c]["frame_force_rmses"] for c in ctypes]
    ax.boxplot(plot_data, tick_labels=ctypes, showfliers=True, whis=(5, 95))
    ax.set_ylabel("per-frame force RMSE (meV/A)")
    ax.set_title("Force RMSE spread by config_type (whiskers = 5th/95th pct)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "force_rmse_boxplot.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Unified MACE evaluation and error distribution visualizer."
    )
    ap.add_argument(
        "--xyz", required=True, help="Path to mace_eval_configs extxyz output"
    )
    ap.add_argument(
        "--out_dir",
        default="./eval_results",
        help="Directory to write output plots and CSVs",
    )
    ap.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="Optional: restrict distribution plots to specific config_type labels",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data(args.xyz)
    build_and_write_summary_tables(data, out_dir)

    print("\nGenerating plots...")
    plot_parity(data, out_dir)
    plot_distributions(
        data, out_dir, types_filter=set(args.types) if args.types else None
    )
    print(f"All plots and tables successfully generated in {out_dir}/")


if __name__ == "__main__":
    main()