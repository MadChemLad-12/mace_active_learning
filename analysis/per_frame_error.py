#!/usr/bin/env python3
"""
Per-frame error ranking for a MACE model, plus removal of chosen frames
from a training set.

MODE 1 - evaluate (needs the model):
    python per_frame_errors.py --model final.model --data training_clean.xyz \
        --energy-key REF_energy --forces-key REF_forces --head Default --top 20
  -> prints global / per-config_type RMSEs, worst-N frames, RMSE-after-removing-k,
     writes <out>.csv and <out>_worst.xyz  (idx = 0-based position in --data)

MODE 2 - remove frames (no model / GPU / mace import needed):
    python per_frame_errors.py --data training_clean.xyz \
        --remove-idx 815 583 510,593 825-827 --out-data training_clean_v2.xyz
  -> writes every frame EXCEPT the listed indices to --out-data, and the removed
     frames (tagged with orig_idx) to <out-data stem>_removed.xyz.
  Indices always refer to the ORIGINAL --data file (0-based, as printed as 'idx').
  Accepted forms: space separated, comma separated, inclusive ranges (a-b).
  Do all removals from the same original file in one go: indices shift in the
  new file.
"""
import argparse
import os
import sys

import numpy as np
from ase.io import read, write


# ----------------------------------------------------------------------------
# Mode 2: remove frames
# ----------------------------------------------------------------------------
def parse_idx(tokens):
    out = []
    for tok in tokens:
        for part in tok.replace(",", " ").split():
            if "-" in part[1:]:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


def remove_frames(frames, remove, data_path, out_path):
    n = len(frames)
    bad = [i for i in remove if i < 0 or i >= n]
    if bad:
        sys.exit(f"ERROR: indices out of range (file has {n} frames, 0-based): {bad}")
    if os.path.abspath(out_path) == os.path.abspath(data_path):
        sys.exit("ERROR: --out-data must differ from --data (refusing to overwrite the input)")

    rm = set(remove)
    kept = [at for i, at in enumerate(frames) if i not in rm]
    removed = []
    print(f"Removing {len(rm)} of {n} frames:")
    for i in remove:
        at = frames[i].copy()
        at.info["orig_idx"] = int(i)
        removed.append(at)
        print(f"  idx {i:5d}  {at.get_chemical_formula():<24s} natoms={len(at):4d}  "
              f"config_type={at.info.get('config_type', 'none')}")

    write(out_path, kept, format="extxyz")
    stem, _ = os.path.splitext(out_path)
    removed_path = f"{stem}_removed.xyz"
    write(removed_path, removed, format="extxyz")

    n_check = len(read(out_path, ":"))
    status = "OK" if n_check == n - len(rm) else "MISMATCH - check the output!"
    print(f"\nWrote {out_path}: {n_check} frames (expected {n - len(rm)}) [{status}]")
    print(f"Wrote {removed_path}: {len(removed)} frames (orig_idx stored in info)")


# ----------------------------------------------------------------------------
# Mode 1: evaluate
# ----------------------------------------------------------------------------
def get_ref(at, e_key, f_key):
    """Read reference energy/forces BEFORE attaching the MACE calculator."""
    e = float(at.info[e_key]) if e_key in at.info else float(at.get_potential_energy())
    f = np.array(at.arrays[f_key]) if f_key in at.arrays else np.array(at.get_forces())
    return e, f


def evaluate(args, frames):
    import pandas as pd
    from mace.calculators import MACECalculator

    kw = dict(model_paths=args.model, device=args.device, default_dtype=args.dtype)
    if args.head:
        kw["head"] = args.head
    calc = MACECalculator(**kw)

    rows = []
    for i, at in enumerate(frames):
        ctype = at.info.get("config_type", "none")
        n = len(at)
        if n == 1 or ctype == "IsolatedAtom":
            continue
        try:
            e_ref, f_ref = get_ref(at, args.energy_key, args.forces_key)
        except Exception as exc:
            print(f"  frame {i}: no reference labels ({exc}), skipped")
            continue
        at.calc = calc
        e = at.get_potential_energy()
        f = at.get_forces()
        d = f - f_ref
        dnorm = np.linalg.norm(d, axis=1)
        w = int(np.argmax(dnorm))
        rows.append(dict(
            idx=i,
            config_type=ctype,
            formula=at.get_chemical_formula(),
            natoms=n,
            dE_per_atom_meV=1000.0 * (e - e_ref) / n,
            F_rmse_meV_A=1000.0 * np.sqrt(np.mean(d ** 2)),
            F_ref_rms_meV_A=1000.0 * np.sqrt(np.mean(f_ref ** 2)),
            F_maxerr_meV_A=1000.0 * dnorm[w],
            worst_atom_idx=w,
            worst_atom_sym=at.get_chemical_symbols()[w],
            _sq=float(np.sum(d ** 2)),
            _ncomp=int(d.size),
        ))
        if len(rows) % 50 == 0:
            print(f"  evaluated {len(rows)} frames...")

    df = pd.DataFrame(rows)
    df["absdE"] = df["dE_per_atom_meV"].abs()
    df["rel_F_pct"] = 100.0 * df["F_rmse_meV_A"] / df["F_ref_rms_meV_A"]

    def global_rmse(d):
        e = np.sqrt(np.mean(d["dE_per_atom_meV"] ** 2))
        f = 1000.0 * np.sqrt(d["_sq"].sum() / d["_ncomp"].sum())
        return e, f

    e_all, f_all = global_rmse(df)
    print(f"\nGLOBAL  RMSE E = {e_all:.2f} meV/atom   RMSE F = {f_all:.2f} meV/A   "
          f"({len(df)} frames)")

    g = df.groupby("config_type")
    summ = pd.DataFrame({
        "n": g.size(),
        "RMSE_E": g["dE_per_atom_meV"].apply(lambda x: np.sqrt(np.mean(x ** 2))),
        "median|dE|": g["absdE"].median(),
        "max|dE|": g["absdE"].max(),
        "RMSE_F": g.apply(lambda d: 1000.0 * np.sqrt(d["_sq"].sum() / d["_ncomp"].sum())),
        "max_F_rmse": g["F_rmse_meV_A"].max(),
    }).sort_values("RMSE_E", ascending=False)
    print("\nPer config_type (large RMSE vs small median => a few outliers dominate):")
    print(summ.round(2).to_string())

    cols = ["idx", "config_type", "formula", "natoms", "dE_per_atom_meV",
            "F_rmse_meV_A", "F_maxerr_meV_A", "worst_atom_sym", "worst_atom_idx", "rel_F_pct"]
    print(f"\nWorst {args.top} by |dE| per atom:")
    worst_e = df.sort_values("absdE", ascending=False).head(args.top)
    print(worst_e[cols].round(2).to_string(index=False))

    print(f"\nWorst {args.top} by force RMSE:")
    worst_f = df.sort_values("F_rmse_meV_A", ascending=False).head(args.top)
    print(worst_f[cols].round(2).to_string(index=False))

    print("\nGlobal RMSE after removing the k worst frames (ranked by |dE|):")
    ranked = df.sort_values("absdE", ascending=False)
    for k in (0, 5, 10, 20, 50):
        if k >= len(ranked):
            continue
        e, f = global_rmse(ranked.iloc[k:])
        print(f"  k={k:3d}:  E = {e:6.2f} meV/atom   F = {f:6.2f} meV/A")

    df.drop(columns=["_sq", "_ncomp"]).to_csv(f"{args.out}.csv", index=False)
    keep = sorted(set(worst_e["idx"]) | set(worst_f["idx"]))
    out_frames = []
    for i in keep:
        at = frames[i].copy()
        r = df[df["idx"] == i].iloc[0]
        at.info["orig_idx"] = int(i)
        at.info["dE_per_atom_meV"] = float(r["dE_per_atom_meV"])
        at.info["F_rmse_meV_A"] = float(r["F_rmse_meV_A"])
        out_frames.append(at)
    write(f"{args.out}_worst.xyz", out_frames)
    print(f"\nWrote {args.out}.csv and {args.out}_worst.xyz "
          f"({len(out_frames)} frames; 'orig_idx' = position in {args.data})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="extxyz with reference labels")
    # evaluate mode
    p.add_argument("--model", help="MACE .model file (evaluate mode)")
    p.add_argument("--energy-key", default="REF_energy")
    p.add_argument("--forces-key", default="REF_forces")
    p.add_argument("--head", default=None, help="only needed for multihead models")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default="frame_errors", help="prefix for evaluate-mode outputs")
    # remove mode
    p.add_argument("--remove-idx", nargs="+", metavar="IDX",
                   help="0-based frame indices in --data to drop (e.g. 815 583 510,593 825-827)")
    p.add_argument("--out-data", help="output extxyz without the removed frames")
    args = p.parse_args()

    if args.remove_idx and not args.out_data:
        p.error("--remove-idx requires --out-data")
    if args.out_data and not args.remove_idx:
        p.error("--out-data only makes sense together with --remove-idx")
    if not args.remove_idx and not args.model:
        p.error("evaluate mode needs --model (or use --remove-idx/--out-data to filter)")

    frames = read(args.data, ":")
    print(f"Read {len(frames)} frames from {args.data}")

    if args.remove_idx:
        remove_frames(frames, parse_idx(args.remove_idx), args.data, args.out_data)
    else:
        evaluate(args, frames)


if __name__ == "__main__":
    main()