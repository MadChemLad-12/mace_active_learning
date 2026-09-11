#!/usr/bin/env python3
"""
check_max_forces.py

Scan an extended XYZ training set and report per-configuration force
statistics (max |F| per atom, mean |F|, force RMS) so you can spot
outlier structures (e.g. unconverged TS/near-TS DFT points) that may
be dominating your MACE force loss.

Usage:
    python check_max_forces.py training_clean.xyz --key REF_forces
    python check_max_forces.py training_clean.xyz --key forces --top 30 --threshold 5.0

Requires: ase  (pip install ase --break-system-packages)
"""

import argparse
import numpy as np
from ase.io import read


def get_force_array(atoms, key):
    """Try the requested key, then fall back to common alternatives."""
    for candidate in (key, "REF_forces", "forces", "force"):
        if candidate in atoms.arrays:
            return atoms.arrays[candidate]
    raise KeyError(
        f"No force array found under '{key}' or common fallbacks "
        f"(REF_forces/forces/force). Available arrays: {list(atoms.arrays.keys())}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xyz_file", help="Path to extended XYZ file")
    p.add_argument("--key", default="REF_forces", help="Force array key to read (default: REF_forces)")
    p.add_argument("--top", type=int, default=20, help="Print the N worst configs by max force (default: 20)")
    p.add_argument("--threshold", type=float, default=None,
                   help="If set, also list every config exceeding this max-force value (eV/A)")
    args = p.parse_args()

    print(f"Reading {args.xyz_file} ...")
    frames = read(args.xyz_file, index=":")
    print(f"Loaded {len(frames)} configurations.\n")

    records = []
    for i, atoms in enumerate(frames):
        try:
            forces = get_force_array(atoms, args.key)
        except KeyError as e:
            print(f"[config {i}] SKIPPED: {e}")
            continue

        mags = np.linalg.norm(forces, axis=1)  # per-atom force magnitude
        fmax = mags.max()
        fmean = mags.mean()
        frms = np.sqrt((mags ** 2).mean())
        natoms = len(atoms)

        # config identifier: try common info keys, else fall back to index
        name = (
            atoms.info.get("config_name")
            or atoms.info.get("head")
            or atoms.info.get("comment")
            or f"config_{i}"
        )

        records.append({
            "index": i,
            "name": name,
            "natoms": natoms,
            "fmax": fmax,
            "fmean": fmean,
            "frms": frms,
        })

    if not records:
        print("No configs with readable forces found — check --key.")
        return

    fmax_all = np.array([r["fmax"] for r in records])
    print("=== Dataset-wide summary ===")
    print(f"  configs analyzed : {len(records)}")
    print(f"  fmax   min/mean/median/max : "
          f"{fmax_all.min():.2f} / {fmax_all.mean():.2f} / {np.median(fmax_all):.2f} / {fmax_all.max():.2f} eV/A")
    for pct in (90, 95, 99):
        print(f"  fmax   {pct}th percentile : {np.percentile(fmax_all, pct):.2f} eV/A")
    print()

    # sort worst-first by max per-atom force
    records.sort(key=lambda r: r["fmax"], reverse=True)

    print(f"=== Top {args.top} configs by max per-atom force ===")
    print(f"{'idx':>6} {'name':<25} {'natoms':>7} {'fmax':>10} {'fmean':>10} {'frms':>10}")
    for r in records[: args.top]:
        print(f"{r['index']:>6} {r['name']:<25} {r['natoms']:>7} "
              f"{r['fmax']:>10.2f} {r['fmean']:>10.2f} {r['frms']:>10.2f}")

    if args.threshold is not None:
        over = [r for r in records if r["fmax"] > args.threshold]
        print(f"\n=== Configs exceeding fmax > {args.threshold} eV/A: {len(over)} ===")
        for r in over:
            print(f"  idx {r['index']:>6}  {r['name']:<25}  fmax={r['fmax']:.2f} eV/A  natoms={r['natoms']}")

        # quick suggestion for a filtered file
        if over:
            print(
                f"\nTip: to write a cleaned dataset excluding these, filter indices "
                f"{{r['index'] for r in records if r['fmax'] <= args.threshold}} "
                f"when re-reading with ase.io.read(..., index=':') and ase.io.write()."
            )


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
check_max_forces.py

Scan an extended XYZ training set and report per-configuration force
statistics (max |F| per atom, mean |F|, force RMS) so you can spot
outlier structures (e.g. unconverged TS/near-TS DFT points) that may
be dominating your MACE force loss.

Usage:
    python check_max_forces.py training_clean.xyz --key REF_forces
    python check_max_forces.py training_clean.xyz --key forces --top 30 --threshold 5.0

Requires: ase  (pip install ase --break-system-packages)
"""

import argparse
import numpy as np
from ase.io import read


def get_force_array(atoms, key):
    """Try the requested key, then fall back to common alternatives."""
    for candidate in (key, "REF_forces", "forces", "force"):
        if candidate in atoms.arrays:
            return atoms.arrays[candidate]
    raise KeyError(
        f"No force array found under '{key}' or common fallbacks "
        f"(REF_forces/forces/force). Available arrays: {list(atoms.arrays.keys())}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xyz_file", help="Path to extended XYZ file")
    p.add_argument("--key", default="REF_forces", help="Force array key to read (default: REF_forces)")
    p.add_argument("--top", type=int, default=20, help="Print the N worst configs by max force (default: 20)")
    p.add_argument("--threshold", type=float, default=None,
                   help="If set, also list every config exceeding this max-force value (eV/A)")
    args = p.parse_args()

    print(f"Reading {args.xyz_file} ...")
    frames = read(args.xyz_file, index=":")
    print(f"Loaded {len(frames)} configurations.\n")

    records = []
    for i, atoms in enumerate(frames):
        try:
            forces = get_force_array(atoms, args.key)
            DFT_forc = get_force_array(atoms, "force")
        except KeyError as e:
            print(f"[config {i}] SKIPPED: {e}")
            continue

        mags = np.linalg.norm(forces, axis=1)  # per-atom force magnitude
        dft_mags = np.linalg.norm(DFT_forc, axis=1)
        fmax = mags.max()
        fmean = mags.mean()
        frms = np.sqrt((mags ** 2).mean())
        rmse = np.sqrt(np.mean((mags - dft_mags)**2)) * 1000
        natoms = len(atoms)

        # config identifier: try common info keys, else fall back to index
        name = (
            atoms.info.get("config_name")
            or atoms.info.get("head")
            or atoms.info.get("comment")
            or f"config_{i}"
        )

        records.append({
            "index": i,
            "name": name,
            "natoms": natoms,
            "fmax": fmax,
            "fmean": fmean,
            "frms": frms,
            "rmse": rmse,
        })

    if not records:
        print("No configs with readable forces found — check --key.")
        return

    fmax_all = np.array([r["fmax"] for r in records])
    print("=== Dataset-wide summary ===")
    print(f"  configs analyzed : {len(records)}")
    print(f"  fmax   min/mean/median/max : "
          f"{fmax_all.min():.2f} / {fmax_all.mean():.2f} / {np.median(fmax_all):.2f} / {fmax_all.max():.2f} eV/A")
    for pct in (90, 95, 99):
        print(f"  fmax   {pct}th percentile : {np.percentile(fmax_all, pct):.2f} eV/A")
    print()

    # sort worst-first by max per-atom force
    records.sort(key=lambda r: r["fmax"], reverse=True)

    print(f"=== Top {args.top} configs by max per-atom force ===")
    print(f"{'idx':>6} {'name':<25} {'natoms':>7} {'fmax':>10} {'fmean':>10} {'frms':>10}")
    for r in records[: args.top]:
        print(f"{r['index']:>6} {r['name']:<25} {r['natoms']:>7} "
              f"{r['fmax']:>10.2f} {r['fmean']:>10.2f} {r['frms']:>10.2f}")

    if args.threshold is not None:
        over = [r for r in records if r["fmax"] > args.threshold]
        print(f"\n=== Configs exceeding fmax > {args.threshold} eV/A: {len(over)} ===")
        for r in over:
            print(f"  idx {r['index']:>6}  {r['name']:<25}  fmax={r['fmax']:.2f} eV/A  natoms={r['natoms']}")

        # quick suggestion for a filtered file
        if over:
            print(
                f"\nTip: to write a cleaned dataset excluding these, filter indices "
                f"{{r['index'] for r in records if r['fmax'] <= args.threshold}} "
                f"when re-reading with ase.io.read(..., index=':') and ase.io.write()."
            )
    



if __name__ == "__main__":
    main()