#!/usr/bin/env python3
"""
compare_models.py
=================
Compare MACE model versions against each other and the foundation model
on a fixed test set.

USAGE
-----
  # First time: carve out a held-out set from your pool (do this ONCE after round 1)
  python compare_models.py --make-held-out

  # Basic — compare all auto-discovered rounds against held-out set
  python compare_models.py --test held_out.xyz

  # Specify models explicitly
  python compare_models.py --test held_out.xyz \
      --models mace-mp-0b3-medium-float32.model \
               mace_V1_active_learning_final.pt \
               mace_V2_active_learning_final.pt \
      --labels Foundation Round_1 Round_2

  # Filter to one system type only
  python compare_models.py --test held_out.xyz --system Dry0.0Pt

OUTPUT
------
  comparison_results/
    metrics_summary.txt          human-readable table
    metrics_summary.csv          machine-readable
    parity_{model}.png           energy + force parity per model
    rmse_progression.png         RMSE vs round — the key improvement plot
    per_system_rmse.png          breakdown by system_type
    error_distributions.png      Probability density of energy & force errors
"""

import os
import re
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read, write
from ase.config import cfg
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from MACE_CP2K_pipeline.src.patches import apply_dftd3_cell_patch
apply_dftd3_cell_patch()

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_TEST_SET = "held_out.xyz"
DEFAULT_OUTPUT   = "comparison_results"
MAX_FORCE_THRESHOLD = 50.0
APPLY_D3 = True  # Whether to include D3 in all calculations (MACE + D3) for this round. If False, only MACE is used.
COLORS = ["#6c757d", "#2196F3", "#4CAF50", "#FF9800",
          "#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]


def autodiscover_models():
    """Find foundation model + all round finals in the current directory."""
    models = []
    for candidate in ["mace-mp-0b3-medium-float32.model",
                      ]:
        #Look into https://huggingface.co/mace-foundations/mace-mh-1 
        if os.path.exists(candidate):
            models.append(("Foundation", candidate))
            break

    for pt in sorted(Path(".").glob("mace_V*_active_learning_final.pt")):
        m = re.search(r"mace_V(\d+)_", pt.name)
        label = f"Round {m.group(1)}" if m else pt.stem
        models.append((label, str(pt)))

    return models

# ==============================================================================
# E0s helpers
# ==============================================================================
 
def parse_e0s(s: str) -> dict:
    """Parse 'H:-13.587 O:-431.601' or JSON '{\"H\":-13.587}' -> {sym: float}."""
    if not s:
        return {}
    s = s.strip()
    if s.startswith("{"):
        import json
        return {k: float(v) for k, v in json.loads(s).items()}
    result = {}
    for token in s.replace(",", " ").split():
        elem, val = token.replace("=", ":").split(":")
        result[elem.strip()] = float(val.strip())
    return result
 
def frame_e0s_shift(atoms, e0s: dict) -> float:
    """Return E_ref = Σ N_i * E0_i for a frame. Returns 0.0 if e0s is empty."""
    if not e0s:
        return 0.0
    from collections import Counter
    counts = Counter(atoms.get_chemical_symbols())
    missing = [el for el in counts if el not in e0s]
    if missing:
        raise ValueError(f"Elements {missing} not in E0s dict {list(e0s.keys())}")
    return sum(n * e0s[el] for el, n in counts.items())
 
def extract_model_e0s(model_path: str) -> dict:
    """
    Read the per-element atomic reference energies stored inside a MACE
    checkpoint.  MACE bakes these in at training time regardless of whether
    you passed --E0s explicitly or used 'average' regression — the fitted
    values always end up in the model file.
 
    Returns {symbol: eV} or {} if extraction fails.
    """
    try:
        import torch
        from ase.data import chemical_symbols
 
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
 
        # MACE saves either the model object directly or a state-dict wrapper
        if hasattr(ckpt, "atomic_energies_fn"):
            ae     = ckpt.atomic_energies_fn.atomic_energies.detach().cpu().tolist()
            z_list = ckpt.atomic_numbers.detach().cpu().tolist()
        elif isinstance(ckpt, dict):
            sd = (ckpt.get("model")
                  or ckpt.get("model_state_dict")
                  or ckpt.get("state_dict")
                  or ckpt)
            ae_key = next(
                (k for k in sd if "atomic_energies" in k and "fn" not in k), None
            )
            if ae_key is None:
                return {}
            ae_tensor = sd[ae_key]
            ae = ae_tensor.detach().cpu().tolist() if hasattr(ae_tensor, "detach") else list(ae_tensor)
            zt     = ckpt.get("z_table") or sd.get("z_table") or {}
            z_list = zt.get("zs", []) if isinstance(zt, dict) else getattr(zt, "zs", [])
        else:
            return {}
 
        if not z_list or len(z_list) != len(ae):
            return {}
 
        return {chemical_symbols[int(z)]: float(e) for z, e in zip(z_list, ae)}
 
    except Exception as exc:
        print(f"    [!] Could not extract E0s from {Path(model_path).name}: {exc}")
        return {}

# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate_model(model_path, test_frames, device="cuda", dtype="float32",
                   ref_e0s=None, pred_e0s=None):
    """
    Evaluate model on every frame.

    ref_e0s  : E0s subtracted from DFT REF_energy  (your CP2K values)
    pred_e0s : E0s subtracted from model prediction (extracted from the
               model checkpoint — whatever it was trained with)

    Subtracting each side's own reference puts both in the same residual
    space, so RMSE is a true accuracy number regardless of model origin.
    Forces are never shifted (∇E0 = 0).
    """
    from mace.calculators import MACECalculator

    ref_e0s  = ref_e0s  or {}
    pred_e0s = pred_e0s or {}

    print(f"  [->] Loading: {Path(model_path).name}")
    calc_mace = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype=dtype
    )
    if APPLY_D3:
        print(f"[→] Including D3 in calculations (MACE + D3)")
        calc_DFT = TorchDFTD3Calculator(
                    device="cuda",
                    damping="bj",
                    xc=cfg.get("dispersion_xc", "pbe"),
                    cutoff=cfg.get("dispersion_cutoff", 40.0),
                )
        calc = SumCalculator([calc_mace, calc_DFT])    
    else:
        print(f"[→] Using MACE only (no D3)")
        calc = calc_mace
    pred_e, ref_e = [], []
    pred_f, ref_f = [], []
    system_types  = []
    failed        = 0

    for i, atoms in enumerate(test_frames):
        if "REF_energy" not in atoms.info:
            failed += 1
            continue

        ref_forces = atoms.arrays["REF_forces"]
        if np.max(np.abs(ref_forces)) > MAX_FORCE_THRESHOLD:
            print(f"    [!] Frame {i} excluded (Max force > {MAX_FORCE_THRESHOLD} eV/A)")
            failed += 1
            continue

        n          = len(atoms)
        dft_shift  = frame_e0s_shift(atoms, ref_e0s)
        pred_shift = frame_e0s_shift(atoms, pred_e0s)
        ref_energy = (atoms.info["REF_energy"] - dft_shift)  / n
        ref_forces = atoms.arrays["REF_forces"]

        try:
            ac = atoms.copy()
            ac.calc = calc
            pred_energy = (ac.get_potential_energy() - pred_shift) / n
            pred_forces = ac.get_forces()

            pred_e.append(pred_energy)
            ref_e.append(ref_energy)
            pred_f.extend(pred_forces.flatten().tolist())
            ref_f.extend(ref_forces.flatten().tolist())
            system_types.append(atoms.info.get("system_type", "unknown"))
        except Exception as ex:
            print(f"    [!] Frame {i} failed: {ex}")
            failed += 1

    if failed:
        print(f"    [!] {failed} frames skipped")

    return {
        "pred_energies": np.array(pred_e),
        "ref_energies":  np.array(ref_e),
        "pred_forces":   np.array(pred_f),
        "ref_forces":    np.array(ref_f),
        "system_types":  system_types,
        "n_frames":      len(pred_e),
        "n_failed":      failed,
    }

def evaluate_per_system(results):
    """Group metrics by system_type using already evaluated model results."""
    by_system = {}
    
    # Unpack pre-calculated values
    for pe, re, pf_chunk, rf_chunk, sys_name in zip(
        results["pred_energies"], 
        results["ref_energies"], 
        # Chunk flattened forces back by atom count per frame
        np.array_split(results["pred_forces"], len(results["pred_energies"])),
        np.array_split(results["ref_forces"], len(results["ref_energies"])),
        results["system_types"]
    ):
        if sys_name not in by_system:
            by_system[sys_name] = {"pe": [], "re": [], "pf": [], "rf": []}
            
        by_system[sys_name]["pe"].append(pe)
        by_system[sys_name]["re"].append(re)
        by_system[sys_name]["pf"].extend(pf_chunk.tolist())
        by_system[sys_name]["rf"].extend(rf_chunk.tolist())

    metrics = {}
    for sys_name, d in by_system.items():
        pe, re = np.array(d["pe"]), np.array(d["re"])
        pf, rf = np.array(d["pf"]), np.array(d["rf"])
        metrics[sys_name] = {
            "energy_rmse_meV_atom": float(np.sqrt(np.mean((pe - re) ** 2))) * 1000,
            "force_rmse_meV_A":     float(np.sqrt(np.mean((pf - rf) ** 2))) * 1000,
            "n_frames": len(pe),
        }
    return metrics

def compute_metrics(r):
    """Global RMSE / MAE / R2 from evaluate_model output."""
    def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
    def mae(a, b):  return float(np.mean(np.abs(a - b)))
    def r2(a, b):
        pred_shifted = a - (np.mean(a) - np.mean(b))
        ss_res = np.sum((b - pred_shifted)**2)
        ss_tot = np.sum((b - np.mean(pred_shifted))**2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    pe, re = r["pred_energies"], r["ref_energies"]
    pf, rf = r["pred_forces"],   r["ref_forces"]
    return {
        "energy_rmse_meV_atom": rmse(pe, re) * 1000,
        "energy_mae_meV_atom":  mae(pe, re)  * 1000,
        "energy_r2":            r2(pe, re),
        "force_rmse_meV_A":     rmse(pf, rf) * 1000,
        "force_mae_meV_A":      mae(pf, rf)  * 1000,
        "force_r2":             r2(pf, rf),
        "n_frames":             r["n_frames"],
    }

# ==============================================================================
# Plots
# ==============================================================================

def plot_parity(results, label, outdir):
    pe, re = results["pred_energies"], results["ref_energies"]
    pf, rf = results["pred_forces"],   results["ref_forces"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Parity — {label}", fontsize=13, fontweight="bold")

    for ax, pred, ref, xl, yl, unit in [
        (axes[0], pe, re,
         "DFT energy (eV/atom)", "MACE energy (eV/atom)", "meV/atom"),
        (axes[1], pf, rf,
         "DFT forces (eV/A)",    "MACE forces (eV/A)",    "meV/A"),
    ]:
        rmse_val = np.sqrt(np.mean((pred - ref)**2)) * 1000
        lims = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
        ax.scatter(ref, pred, s=8, alpha=0.4, color="#2196F3", edgecolors="none")
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.text(0.05, 0.93, f"RMSE = {rmse_val:.1f} {unit}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))

    plt.tight_layout()
    fname = os.path.join(outdir, f"parity_{label.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] {fname}")


def plot_rmse_progression(all_metrics, labels, outdir):
    """Bar chart of energy + force RMSE across rounds — the headline plot."""
    e_rmse = [m["energy_rmse_meV_atom"] for m in all_metrics]
    f_rmse = [m["force_rmse_meV_A"]     for m in all_metrics]
    x      = range(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Improvement Across Rounds", fontsize=13, fontweight="bold")

    for ax, vals, title, ylabel in [
        (ax1, e_rmse, "Energy RMSE", "RMSE (meV/atom)"),
        (ax2, f_rmse, "Force RMSE",  "RMSE (meV/A)"),
    ]:
        bars = ax.bar(x, vals, color=COLORS[:len(labels)],
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(outdir, "rmse_progression.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] {fname}")


def plot_per_system(per_system_all, labels, outdir):
    """Grouped bar chart: force RMSE per system per model version."""
    all_systems = sorted({s for ps in per_system_all for s in ps.keys()})
    if not all_systems:
        return

    n_sys    = len(all_systems)
    n_models = len(labels)
    x        = np.arange(n_sys)
    width    = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(12, n_sys * 2), 6))

    for mi, (label, ps) in enumerate(zip(labels, per_system_all)):
        vals   = [ps.get(s, {}).get("force_rmse_meV_A", 0) for s in all_systems]
        offset = (mi - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label,
               color=COLORS[mi % len(COLORS)], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(all_systems, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Force RMSE (meV/A)", fontsize=10)
    ax.set_title("Per-System Force RMSE Across Model Versions",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(outdir, "per_system_rmse.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] {fname}")

def plot_error_distributions(all_results, labels, outdir):
    """
    Plots the probability density function (distribution) of errors 
    for both energy (meV/atom) and forces (meV/A) for all models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Error Distributions Relative to CP2K Reference Data", fontsize=14, fontweight="bold")

    for i, (label, res) in enumerate(zip(labels, all_results)):
        color = COLORS[i % len(COLORS)]
        
        # 1. Energy Error Calculations (meV/atom)
        e_error = (res["pred_energies"] - res["ref_energies"]) * 1000
        # 2. Force Error Calculations (meV/A)
        f_error = (res["pred_forces"] - res["ref_forces"]) * 1000

        # Plot Energy Distribution
        ax1.hist(e_error, bins=50, density=True, histtype="step", linewidth=2,
                 color=color, label=label, alpha=0.85)
        
        # Plot Force Distribution
        ax2.hist(f_error, bins=50, density=True, histtype="step", linewidth=2,
                 color=color, label=label, alpha=0.85)

    # Styling Energy Axis
    ax1.axvline(0, color="black", linestyle="--", alpha=0.5, lw=1)
    ax1.set_xlabel("Energy Error ($E_{MACE} - E_{REF}$) [meV/atom]", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.set_title("Energy Error Distribution", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=9)

    # Styling Force Axis
    ax2.axvline(0, color="black", linestyle="--", alpha=0.5, lw=1)
    ax2.set_xlabel("Force Error ($F_{MACE} - F_{REF}$) [meV/$\AA$]", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.set_title("Force Error Distribution", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(outdir, "error_distributions.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] {fname}")

# ==============================================================================
# Summary table
# ==============================================================================

def print_and_save_summary(all_metrics, labels, outdir):
    headers = ["Model", "E-RMSE(meV/at)", "E-MAE", "E-R2",
               "F-RMSE(meV/A)", "F-MAE", "F-R2", "Frames"]
    col = 18

    sep   = "=" * (col * len(headers))
    lines = [sep, "  MACE MODEL COMPARISON", sep,
             "  " + "  ".join(f"{h:<{col}}" for h in headers),
             "-" * (col * len(headers))]

    csv_rows = [",".join(headers)]
    for label, m in zip(labels, all_metrics):
        row = [label,
               f"{m['energy_rmse_meV_atom']:.2f}",
               f"{m['energy_mae_meV_atom']:.2f}",
               f"{m['energy_r2']:.4f}",
               f"{m['force_rmse_meV_A']:.2f}",
               f"{m['force_mae_meV_A']:.2f}",
               f"{m['force_r2']:.4f}",
               str(m["n_frames"])]
        lines.append("  " + "  ".join(f"{v:<{col}}" for v in row))
        csv_rows.append(",".join(row))

    lines.append(sep)
    summary = "\n".join(lines)
    print("\n" + summary)

    with open(os.path.join(outdir, "metrics_summary.txt"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(outdir, "metrics_summary.csv"), "w") as f:
        f.write("\n".join(csv_rows) + "\n")

    print(f"\n[+] Summary saved to {outdir}/")

# ==============================================================================
# Held-out set helper
# ==============================================================================
def create_held_out_set(
    pool_file="training_clean.xyz",
    held_out_file="held_out.xyz",
    cleaned_pool_file="train_compare.xyz",
    n=50,
    seed=42,
):
    """
    Stratified sample from the raw pool BEFORE cleaning/training.
    Removes held-out frames from the pool so they never enter training.
    
    Call ONCE before round 1, or before round 5 if starting fresh.
    Never call again — the guard clause prevents overwriting.

    Args:
        pool_file:         Source of all DFT frames (master pool).
        held_out_file:     Where to write the held-out frames.
        cleaned_pool_file: The file passed to MACE training — held-out
                           frames are excluded from this.
        n:                 Total held-out frames to draw.
        seed:              Random seed for reproducibility.
    """
    # ----------------------------------------------------------------
    # Guard: never overwrite an existing held-out set
    # ----------------------------------------------------------------
    from collections import defaultdict
    import random

    if os.path.exists(held_out_file):
        print(f"[!] {held_out_file} already exists — not overwriting.")
        print(f"    Delete it manually if you want to recreate it.")
        return

    if not os.path.exists(pool_file):
        print(f"[X] Pool file not found: {pool_file}")
        return

    from collections import defaultdict
    import random
    random.seed(seed)

    # ----------------------------------------------------------------
    # Load and stratify by system_type
    # ----------------------------------------------------------------
    all_frames = read(pool_file, index=":")
    
    # Tag each frame with its index so we can exclude them later
    for i, atoms in enumerate(all_frames):
        atoms.info["_pool_index"] = i

    by_system = defaultdict(list)
    for atoms in all_frames:
        key = atoms.info.get("system_type", "unknown")
        by_system[key].append(atoms)

    # ----------------------------------------------------------------
    # Stratified sampling — equal budget per system type
    # ----------------------------------------------------------------
    n_systems  = len(by_system)
    per_system = max(1, n // n_systems)
    
    held_out      = []
    held_out_idxs = set()

    for sys_name, sys_frames in sorted(by_system.items()):
        k = min(per_system, len(sys_frames))
        chosen = random.sample(sys_frames, k)
        held_out.extend(chosen)
        held_out_idxs.update(a.info["_pool_index"] for a in chosen)
        print(f"  [{sys_name}]  {k}/{len(sys_frames)} frames selected for held-out")

    # ----------------------------------------------------------------
    # Write held-out set
    # ----------------------------------------------------------------
    write(held_out_file, held_out, format="extxyz")
    print(f"\n[+] Held-out set written: {held_out_file}  "
          f"({len(held_out)} frames across {n_systems} system types)")

    # ----------------------------------------------------------------
    # Write the training pool with held-out frames REMOVED
    # ----------------------------------------------------------------
    training_frames = [
        a for a in all_frames
        if a.info["_pool_index"] not in held_out_idxs
    ]

    # Clean up the temporary index tag before writing
    for atoms in training_frames:
        atoms.info.pop("_pool_index", None)
    for atoms in held_out:
        atoms.info.pop("_pool_index", None)

    write(cleaned_pool_file, training_frames, format="extxyz")
    print(f"[+] Training pool written: {cleaned_pool_file}  "
          f"({len(training_frames)} frames, held-out excluded)")
    print(f"\n[!] IMPORTANT: Add new DFT frames to {pool_file} each round,")
    print(f"    then re-run your cleaning pipeline — held-out frames will")
    print(f"    be automatically excluded via the index guard.")

# ==============================================================================
# Main
# ==============================================================================
def main():
    # ------------------------------------------------------------------
    # CP2K isolated-atom reference energies (eV) — your DFT baseline.
    # These are ALWAYS subtracted from REF_energy (the DFT side).
    # Keyed by atomic number for easy cross-checking with CP2K output.
    # ------------------------------------------------------------------
    from ase.data import chemical_symbols as _cs
    import json

    # You can add your own below if you like I 
    E0_JSON = "E0s.json"
    try: 
        with open(E0_JSON, "r") as file:
            E0s_ref = {int(k): v for k, v in json.load(file).items()}
            print(f"Your E0s are {E0s_ref}")
    except FileNotFoundError:
        print(f"Error: The file '{E0_JSON}' could not be found.")
        E0s_ref = {}
        print(f"The energy will not be reliable so only read forces")

        
    CP2K_E0S = E0s_ref
 
    parser = argparse.ArgumentParser(
        description="Compare MACE model versions on a fixed test set"
    )
    parser.add_argument("--test",   default=DEFAULT_TEST_SET)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model file paths (auto-discovered if omitted)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each model")
    parser.add_argument("--system", default=None,
                        help="Filter to one system_type")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--make-held-out", action="store_true",
                        help="Create held_out.xyz from cleaned data then exit")
    parser.add_argument("--cleaned-file", default="cleaned_pool.xyz",
                        help="Cleaned XYZ file to split held-out data from")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.make_held_out:
        create_held_out_set()
        return

    if not os.path.exists(args.test):
        print(f"[X] Test set not found: {args.test}")
        print(f"    Run first:  python compare_models.py --make-held-out")
        sys.exit(1)

    print(f"[->] Loading test set: {args.test}")
    test_frames = read(args.test, index=":")

    if args.system:
        test_frames = [f for f in test_frames
                       if f.info.get("system_type") == args.system]
        print(f"[->] Filtered to '{args.system}': {len(test_frames)} frames")

    print(f"[+] Test frames: {len(test_frames)}")
    print(f"\n[E0s] DFT reference (CP2K) — applied to REF_energy for all models:")
    for sym, val in sorted(CP2K_E0S.items()):
        print(f"      {sym}: {val:.4f} eV")
    print()

    # Resolve model list
    if args.models:
        model_list = list(zip(
            args.labels if args.labels else [Path(m).stem for m in args.models],
            args.models
        ))
    else:
        model_list = autodiscover_models()
        if not model_list:
            print("[X] No models found. Use --models to specify them.")
            sys.exit(1)

    print(f"[->] Models ({len(model_list)}):")
    for label, path in model_list:
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"    [{status}] {label:<20}  {path}")
    print()

    all_metrics    = []
    all_per_system = []
    all_results    = []
    labels         = []

    for label, model_path in model_list:
        if not os.path.exists(model_path):
            print(f"[!] Skipping {label} — not found")
            continue

        print(f"[->] Evaluating: {label}")

        # ---------------------------------------------------------------
        # Always extract E0s from the checkpoint for the PREDICTION side.
        #
        # Every MACE model — yours or the foundation — stores the E0s it
        # was trained with inside the .model/.pt file.  Using those values
        # (not your CP2K values) to shift the prediction side means we are
        # always comparing true residuals, regardless of whether the model
        # used CP2K E0s, regression-fitted E0s, or Materials Project E0s.
        #
        # DFT side always uses CP2K_E0S (your data's reference frame).
        # ---------------------------------------------------------------
        pred_e0s = extract_model_e0s(model_path)
        if pred_e0s:
            print(f"  [E0s] Prediction side — extracted from checkpoint:")
            for sym, val in sorted(pred_e0s.items()):
                print(f"        {sym}: {val:.4f} eV")
        else:
            print(f"  [E0s] WARNING: could not extract E0s from checkpoint.")
            print(f"        Prediction side will use no shift — energy RMSE")
            print(f"        may be inflated by a reference-frame offset.")
            print(f"        Using default E0s")
            pred_e0s=CP2K_E0S
 
        results = evaluate_model(
            model_path, test_frames, device=args.device,
            ref_e0s=CP2K_E0S, pred_e0s=pred_e0s,
        )
        metrics = compute_metrics(results)
        per_system = evaluate_per_system(results)

        all_metrics.append(metrics)
        all_per_system.append(per_system)
        all_results.append(results)
        labels.append(label)

        print(f"  [+] E-RMSE: {metrics['energy_rmse_meV_atom']:.2f} meV/atom  "
              f"F-RMSE: {metrics['force_rmse_meV_A']:.2f} meV/A  "
              f"({metrics['n_frames']} frames)\n")

        plot_parity(results, label, args.outdir)

    if not all_metrics:
        print("[X] No models evaluated.")
        sys.exit(1)

    print_and_save_summary(all_metrics, labels, args.outdir)
    plot_rmse_progression(all_metrics, labels, args.outdir)
    plot_per_system(all_per_system, labels, args.outdir)
    plot_error_distributions(all_results, labels, args.outdir)
    
    print(f"\n[+] All outputs in: {args.outdir}/")
    print(f"\n  Key files to check:")
    print(f"  {args.outdir}/rmse_progression.png  <- is the model improving?")
    print(f"  {args.outdir}/per_system_rmse.png   <- which systems need more data?")
    print(f"  {args.outdir}/metrics_summary.txt   <- headline numbers")

if __name__ == "__main__":
    main()