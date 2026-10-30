#!/usr/bin/env python3
"""
Evaluate a MACE model on a labelled test set and analyse the errors.

Mode 1: generate predictions in Python (calls mace.cli.eval_configs) and analyse them
    python mace_eval_analysis.py --test_xyz test.xyz --model ft.model --head Default --outdir out_ft
    python mace_eval_analysis.py --test_xyz test.xyz --model mace-mp-0b3-medium.model \
        --is_foundation --outdir out_foundation

Mode 2: analyse an existing mace_eval_configs output
    python mace_eval_analysis.py --eval_xyz test_pred.xyz --outdir out_ft

Foundation models are compared after removing the E0 / DFT-reference offset
(--shift_mode element = per-element least-squares fit, --shift_mode e0 = exact per-element
E0 difference using --e0_json, --shift_mode mean = one constant shift of the total energy).
Fine-tuned models are compared with no shift.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless (HPC) safe
import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from ase.io import read

ENERGY_THRESHOLD_MEV = 10.0  # meV/atom
FORCE_THRESHOLD_MEV = 50.0   # meV/A


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a MACE model on a test set and analyse errors.")
    p.add_argument("--test_xyz", type=str, help="Labelled test set (extxyz) to evaluate the model on")
    p.add_argument("--model", type=str, help="Path to a .model file; if given, predictions are generated here")
    p.add_argument("--eval_xyz", type=str, help="Existing mace_eval_configs output to analyse (skips evaluation)")
    p.add_argument("--outdir", type=str, required=True, help="Where to place output files")
    p.add_argument("--is_foundation", action="store_true",
                   help="Model is a foundation model: remove the energy offset before computing errors")
    p.add_argument("--align", action="store_true",
                   help="Also remove the energy offset for a NON-foundation model (like-for-like comparison "
                        "with a shifted foundation baseline, or to find where a constant offset comes from). "
                        "With --shift_mode e0 the 'source' E0s are the ones stored in --model")
    p.add_argument("--shift_mode", choices=["element", "mean", "e0"], default="element",
                   help="Offset removal for --is_foundation: element = per-element least-squares fit "
                        "(default), mean = constant total shift, e0 = exact E0 difference")
    p.add_argument("--e0_json", type=str,
                   help="E0s of your CP2K labels / fine-tuned model: {atomic number: eV}. Needed for "
                        "--shift_mode e0; for any model it also enables the atomization-energy report")
    p.add_argument("--foundation_e0_json", type=str,
                   help="(shift_mode e0) foundation model E0s as json; alternative to reading them from "
                        "--foundation_model / --model")
    p.add_argument("--foundation_model", type=str,
                   help="(shift_mode e0) .model file to read the foundation E0s from "
                        "(defaults to --model)")
    p.add_argument("--head", type=str, default=None, help="Head name for multihead models (e.g. Default)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--default_dtype", type=str, default="float64")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--info_prefix", type=str, default="MACE_", help="Prefix of predicted keys in the eval file")
    p.add_argument("--ref_energy_key", type=str, default="REF_energy")
    p.add_argument("--ref_forces_key", type=str, default="REF_forces")
    p.add_argument("--energy_threshold", type=float, default=ENERGY_THRESHOLD_MEV, help="RMSE limit, meV/atom")
    p.add_argument("--force_threshold", type=float, default=FORCE_THRESHOLD_MEV, help="RMSE limit, meV/A")
    p.add_argument("--n_worst", type=int, default=10)
    args = p.parse_args()

    if args.eval_xyz and (args.model or args.test_xyz):
        p.error("use either --eval_xyz, or --test_xyz together with --model, not both")
    if not args.eval_xyz and not (args.model and args.test_xyz):
        p.error("provide --eval_xyz, or both --test_xyz and --model")
    if (args.is_foundation or args.align) and args.shift_mode == "e0":
        if not args.e0_json:
            p.error("--shift_mode e0 needs --e0_json")
        if not (args.foundation_e0_json or args.foundation_model or args.model):
            p.error("--shift_mode e0 needs --foundation_e0_json, --foundation_model or --model")
    return args


def run_mace_eval(test_xyz, model, output, device, dtype, batch_size, head, info_prefix):
    """Call mace.cli.eval_configs from Python.

    Its main() parses sys.argv, so we set sys.argv temporarily. Check the flag names with
    `mace_eval_configs --help` if your MACE version differs.
    """
    from mace.cli import eval_configs

    argv = ["mace_eval_configs", "--configs", test_xyz, "--model", model, "--output", output,
            "--device", device, "--default_dtype", dtype, "--batch_size", str(batch_size),
            "--info_prefix", info_prefix]
    if head:
        argv += ["--head", head]

    old_argv = sys.argv
    sys.argv = argv
    try:
        eval_configs.main()
    finally:
        sys.argv = old_argv


def get_key(atoms, key):
    """Look a key up in atoms.info, then atoms.arrays, then the attached calculator."""
    if key in atoms.info:
        return np.asarray(atoms.info[key], dtype=float)
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key], dtype=float)
    raise KeyError(key)


def load_data(path, ref_e_key, ref_f_key, prefix):
    frames = read(path, index=":")
    try:
        dft_e = np.array([float(get_key(a, ref_e_key)) for a in frames])
        dft_f = [get_key(a, ref_f_key) for a in frames]
    except KeyError as e:
        sys.exit(f"Reference key {e} not found in {path}. Check --ref_energy_key / --ref_forces_key "
                 f"(example frame has info={list(frames[0].info)}, arrays={list(frames[0].arrays)})")
    try:
        mace_e = np.array([float(get_key(a, prefix + "energy")) for a in frames])
        mace_f = [get_key(a, prefix + "forces") for a in frames]
    except KeyError as e:
        sys.exit(f"Prediction key {e} not found in {path}. Check --info_prefix.")
    return frames, dft_e, dft_f, mace_e, mace_f


def composition_matrix(frames):
    """Atom counts per element: returns (sorted atomic numbers, counts [n_frames, n_elements])."""
    zs = sorted({int(z) for a in frames for z in a.get_atomic_numbers()})
    counts = np.array([[int(np.sum(a.get_atomic_numbers() == z)) for z in zs] for a in frames], float)
    return zs, counts


def load_e0_json(path):
    """E0 json: {"<atomic number>": energy_in_eV, ...}"""
    with open(path) as fh:
        return {int(k): float(v) for k, v in json.load(fh).items()}


def load_model_e0s(model_path, head=None):
    """Read the atomic energies stored inside a MACE .model file."""
    import torch

    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch has no weights_only argument
        model = torch.load(model_path, map_location="cpu")
    zs = [int(z) for z in model.atomic_numbers]
    e0 = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    if e0.ndim == 2:  # [n_heads, n_elements] in recent MACE versions
        heads = list(getattr(model, "heads", []))
        e0 = e0[heads.index(head) if head in heads else 0]
    return {z: float(e) for z, e in zip(zs, e0)}


def align_energies(frames, dft_e, mace_e, mode, e0_target=None, e0_source=None):
    """Return (adjusted MACE total energies, info about the shift).

    mode "mean"   : one constant shift of the total energy (only valid for fixed composition)
    mode "element": per-element least-squares fit of DFT - MACE (absorbs any systematic offset)
    mode "e0"     : exact per-element shift e0_target[Z] - e0_source[Z] (no test labels used);
                    the per-element fit is also computed, purely as a diagnostic
    """
    delta = dft_e - mace_e
    n_atoms = np.array([len(a) for a in frames])

    if mode == "mean":
        if len(set(n_atoms)) > 1:
            print("WARNING: atom counts vary; a constant total-energy shift is not appropriate. "
                  "Use --shift_mode element or e0.")
        shift = float(np.mean(delta))
        return mace_e + shift, {"mode": "mean", "shift_total_eV": shift}

    zs, counts = composition_matrix(frames)
    syms = [chemical_symbols[z] for z in zs]
    rank = int(np.linalg.matrix_rank(counts))
    fit, *_ = np.linalg.lstsq(counts, delta, rcond=None)
    n_distinct = len({tuple(r) for r in counts})
    if rank < len(zs):
        print(f"WARNING: composition matrix rank {rank} < {len(zs)} elements; fitted per-element shifts "
              "are not unique (the energy residuals are still valid).")
    if mode == "element" and rank >= n_distinct:
        print(f"WARNING: only {n_distinct} distinct composition(s) and the fit has rank {rank}: the shift "
              "absorbs the mean offset of every composition, so the energy errors below only measure "
              "scatter WITHIN each composition. Prefer --shift_mode e0.")

    if mode == "element":
        return mace_e + counts @ fit, {"mode": "element", "rank": rank,
                                       "shift_eV": {s: float(v) for s, v in zip(syms, fit)}}

    # ---- mode "e0" ----
    missing = [chemical_symbols[z] for z in zs if z not in e0_target or z not in e0_source]
    if missing:
        sys.exit(f"E0 missing for element(s) {missing} in the target or source E0 set.")
    e0_diff = np.array([e0_target[z] - e0_source[z] for z in zs])
    adjusted = mace_e + counts @ e0_diff

    if rank >= len(zs):
        print("  per-element shift: exact E0 difference vs fitted (eV)")
        for s, d, f in zip(syms, e0_diff, fit):
            print(f"    {s:3s} E0 diff {d:14.4f}   fitted {f:14.4f}   fitted - E0 diff {f - d:10.4f}")
    else:
        print("  fitted-vs-E0 table skipped: the per-element fit is under-determined for this test set "
              f"(rank {rank} < {len(zs)} elements), so the fitted numbers carry no information.")
    resid_pa = (dft_e - adjusted) / n_atoms * 1000
    groups = defaultdict(list)
    for i, a in enumerate(frames):
        groups[a.get_chemical_formula()].append(i)
    print("  residual (DFT - MACE after E0 alignment) per composition:")
    for formula, idx in groups.items():
        r = (dft_e - adjusted)[idx]
        print(f"    n={len(idx):3d}  mean {np.mean(r):12.3f} eV = {np.mean(resid_pa[idx]):10.1f} meV/atom   {formula}")
    print(f"  mean residual after E0 alignment: {np.mean(resid_pa):.2f} meV/atom "
          f"(std {np.std(resid_pa):.2f})")
    return adjusted, {
        "mode": "e0", "rank": rank,
        "e0_difference_eV": {s: float(v) for s, v in zip(syms, e0_diff)},
        "fitted_shift_eV": {s: float(v) for s, v in zip(syms, fit)},
        "mean_residual_after_e0_mev_per_atom": float(np.mean(resid_pa)),
    }


def r2_score(y, yhat):
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def error_stats(ref, pred, scale=1000.0):
    err = pred - ref
    return {"mae": float(np.mean(np.abs(err)) * scale),
            "rmse": float(np.sqrt(np.mean(err ** 2)) * scale),
            "r2": r2_score(ref, pred)}


def scatter_plot(ref, pred, title_model, metric, units, err_units, outdir, tag, stats, passed, color):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ref, pred, alpha=0.5, color=color, s=15 if len(ref) < 5000 else 2, rasterized=True)
    lo, hi = min(ref.min(), pred.min()), max(ref.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.7, label="Ideal")
    ax.set_xlabel(f"DFT reference ({units})")
    ax.set_ylabel(f"MACE prediction ({units})")
    status = "Passed threshold" if passed else "Failed threshold"
    ax.set_title(f"{title_model} - {metric}\n{status}",
                 color="green" if passed else "red", fontweight="bold")
    text = (f"MAE: {stats['mae']:.2f} {err_units}\nRMSE: {stats['rmse']:.2f} {err_units}\n"
            f"R2: {stats['r2']:.4f}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.3))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fname = f"Scatter_{metric.lower().replace(' ', '_')}_{tag}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tag = "foundation" if args.is_foundation else "finetune"

    if args.model:
        eval_xyz = os.path.join(args.outdir, f"eval_{tag}.xyz")
        print(f"Evaluating {args.model} on {args.test_xyz} ...")
        run_mace_eval(args.test_xyz, args.model, eval_xyz, args.device, args.default_dtype,
                      args.batch_size, args.head, args.info_prefix)
    else:
        eval_xyz = args.eval_xyz

    print(f"Parsing evaluated structures: {eval_xyz}")
    frames, dft_e, dft_f, mace_e, mace_f = load_data(
        eval_xyz, args.ref_energy_key, args.ref_forces_key, args.info_prefix)
    n_atoms = np.array([len(a) for a in frames])

    if args.is_foundation or args.align:
        print("Removing energy offset ...")
        e0_target = e0_source = None
        if args.shift_mode == "e0":
            e0_target = load_e0_json(args.e0_json)
            if args.foundation_e0_json:
                e0_source = load_e0_json(args.foundation_e0_json)
            else:
                e0_source = load_model_e0s(args.foundation_model or args.model, args.head)
        mace_e_adj, shift_info = align_energies(frames, dft_e, mace_e, args.shift_mode,
                                                e0_target, e0_source)
        print(f"  shift: {shift_info}")
    else:
        mace_e_adj, shift_info = mace_e, {"mode": "none"}

    # ---- global scores ----
    dft_e_pa, mace_e_pa = dft_e / n_atoms, mace_e_adj / n_atoms
    dft_f_flat = np.concatenate([f.ravel() for f in dft_f])
    mace_f_flat = np.concatenate([f.ravel() for f in mace_f])

    e_stats = error_stats(dft_e_pa, mace_e_pa)   # meV/atom
    f_stats = error_stats(dft_f_flat, mace_f_flat)  # meV/A
    # atomization energy per atom = (E - sum E0)/N: removes the huge composition spread of absolute
    # energies, so R2 means something and the E0 json can be sanity-checked against the DFT labels
    ae_stats = None
    if args.e0_json:
        e0_cp2k = load_e0_json(args.e0_json)
        zs, counts = composition_matrix(frames)
        miss = [chemical_symbols[z] for z in zs if z not in e0_cp2k]
        if miss:
            sys.exit(f"--e0_json has no E0 for element(s) {miss}")
        e0_vec = np.array([e0_cp2k[z] for z in zs])
        ref_ae = (dft_e - counts @ e0_vec) / n_atoms
        pred_ae = (mace_e_adj - counts @ e0_vec) / n_atoms
        ae_stats = error_stats(ref_ae, pred_ae)
        ae_stats["ref_std_eV_per_atom"] = float(np.std(ref_ae))
        print(f"\nAtomization energy (E - sum E0)/N, reference values per composition:")
        by_formula = defaultdict(list)
        for i, a in enumerate(frames):
            by_formula[a.get_chemical_formula()].append(i)
        for formula, idx in by_formula.items():
            print(f"  n={len(idx):3d}  DFT {np.mean(ref_ae[idx]):8.3f}  MACE {np.mean(pred_ae[idx]):8.3f} eV/atom   {formula}")
        print("  (typical molecular/ionic systems sit at roughly -3 to -6 eV/atom; values far outside "
              "that range point to a bad E0 for one of the elements in that composition)")

    energy_pass = e_stats["rmse"] <= args.energy_threshold
    force_pass = f_stats["rmse"] <= args.force_threshold
    usable = bool(energy_pass and force_pass)

    # ---- per-structure errors ----
    rows = []
    for i, atoms in enumerate(frames):
        rows.append({
            "index": i,
            "config_type": atoms.info.get("config_type", atoms.get_chemical_formula()),
            "n_atoms": int(n_atoms[i]),
            "energy_error_mev_per_atom": float((mace_e_pa[i] - dft_e_pa[i]) * 1000),
            "force_rmse_mev": float(np.sqrt(np.mean((mace_f[i] - dft_f[i]) ** 2)) * 1000),
        })

    # all frames (handy for choosing what to add in the next active-learning round)
    with open(os.path.join(args.outdir, f"per_frame_errors_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    worst_energy = sorted(rows, key=lambda r: abs(r["energy_error_mev_per_atom"]), reverse=True)
    worst_force = sorted(rows, key=lambda r: r["force_rmse_mev"], reverse=True)
    fields = ["ranking_type", "rank"] + list(rows[0].keys())
    with open(os.path.join(args.outdir, f"error_outliers_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rank, item in enumerate(worst_energy[:args.n_worst], 1):
            w.writerow({**item, "ranking_type": "Worst Energy Errors", "rank": rank})
        for rank, item in enumerate(worst_force[:args.n_worst], 1):
            w.writerow({**item, "ranking_type": "Worst Force Errors", "rank": rank})

    # ---- per config_type breakdown ----
    groups = defaultdict(list)
    for r in rows:
        groups[r["config_type"]].append(r)
    by_type = {}
    for ct, rs in groups.items():
        e = np.array([r["energy_error_mev_per_atom"] for r in rs])
        f = np.array([r["force_rmse_mev"] for r in rs])
        by_type[ct] = {"n": len(rs),
                       "energy_bias_mev_per_atom": float(np.mean(e)),
                       "energy_rmse_mev_per_atom": float(np.sqrt(np.mean(e ** 2))),
                       "force_rmse_mev": float(np.sqrt(np.mean(f ** 2)))}

    # ---- plots ----
    color = "#ff7f0e" if args.is_foundation else "#1f77b4"
    title_model = "Foundation" if args.is_foundation else "Fine-tuned"
    if args.is_foundation or args.align:
        title_model += " (offset removed)"
    scatter_plot(dft_e_pa, mace_e_pa, title_model, "Energy per Atom", "eV/atom", "meV/atom",
                 args.outdir, tag, e_stats, energy_pass, color)
    scatter_plot(dft_f_flat, mace_f_flat, title_model, "Force Components", "eV/A", "meV/A",
                 args.outdir, tag, f_stats, force_pass, color)

    # ---- summary ----
    summary = {
        "model": args.model, "eval_xyz": eval_xyz, "n_frames": len(frames),
        "is_foundation": args.is_foundation, "aligned": bool(args.is_foundation or args.align),
        "shift": shift_info,
        "energy": {**e_stats, "units": "meV/atom (MAE, RMSE)", "threshold_rmse": args.energy_threshold,
                   "pass": bool(energy_pass)},
        "forces": {**f_stats, "units": "meV/A (MAE, RMSE)", "threshold_rmse": args.force_threshold,
                   "pass": bool(force_pass)},
        "atomization_energy": ae_stats,
        "usable": usable, "by_config_type": by_type,
    }
    with open(os.path.join(args.outdir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nFrames: {len(frames)}")
    print(f"Energy: MAE {e_stats['mae']:.2f}  RMSE {e_stats['rmse']:.2f} meV/atom  "
          f"R2 {e_stats['r2']:.4f}  ({'PASS' if energy_pass else 'FAIL'} <= {args.energy_threshold})")
    print("  (energy R2 above is on absolute per-atom energies and is dominated by composition spread)")
    if ae_stats:
        print(f"Atomization energy: MAE {ae_stats['mae']:.2f}  RMSE {ae_stats['rmse']:.2f} meV/atom  "
              f"R2 {ae_stats['r2']:.4f}  (std of reference {ae_stats['ref_std_eV_per_atom']:.3f} eV/atom)")
    print(f"Forces: MAE {f_stats['mae']:.1f}  RMSE {f_stats['rmse']:.1f} meV/A  "
          f"R2 {f_stats['r2']:.4f}  ({'PASS' if force_pass else 'FAIL'} <= {args.force_threshold})")
    print(f"Usable: {usable}")
    if len(by_type) > 1:
        print("\nPer config_type:")
        for ct, s in sorted(by_type.items(), key=lambda kv: -kv[1]["force_rmse_mev"]):
            print(f"  {ct:30s} n={s['n']:4d}  E bias {s['energy_bias_mev_per_atom']:+8.2f}  "
                  f"RMSE {s['energy_rmse_mev_per_atom']:7.2f} meV/atom  "
                  f"F {s['force_rmse_mev']:7.1f} meV/A")


if __name__ == "__main__":
    main()