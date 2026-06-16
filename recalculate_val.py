#!/usr/bin/env python3
"""
recalculate_validation.py

Recalculates DFT single points for a validation xyz using the production
CP2K setup from active_pipeline.py. Results are written to a new XYZ
and NEVER added to the training pool.

Usage:
    # Step 1 — write inputs and generate submit script
    python recalculate_validation.py --write

    # Step 2 — run CP2K (on HPC: edit VAL/submit_all.sh first)
    bash VAL/submit_all.sh

    # Step 3 — parse outputs to XYZ
    python recalculate_validation.py --parse

    # Or all steps locally (only if CP2K is available on this machine)
    python recalculate_validation.py --all
"""

import os
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write

# ─────────────────────────────────────────────────────────────────────────────
#  Import production functions from your pipeline
#  These carry your CP2K_TEMPLATE, KIND_PARAMS, constants etc. automatically
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Adds the current 'active_learning' directory to your path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import directly since it's in the same folder
from active_pipeline import (
    write_cp2k_sp,
    parse_cp2k_sp_results,
    parse_stress_from_out,
    _cp2k_output_is_complete,
    _write_submission_script,
    CP2K_TIMEOUT,
    HA_TO_EV,
    HA_BOHR_TO_EV_ANG,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — only things specific to the validation recalculation
# ─────────────────────────────────────────────────────────────────────────────

INPUT_XYZ     = "validate.xyz"
OUTPUT_XYZ    = "validate_corrected.xyz"
VAL_DIR       = Path("VAL")
JOB_PREFIX    = "val"

# Override the ROUND used in job naming so val jobs are clearly separate
# from your training rounds and never accidentally merged
VAL_ROUND_TAG = "val"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Write CP2K inputs
# ─────────────────────────────────────────────────────────────────────────────

def step_write(frames):
    """Write one CP2K .inp per frame using the production write_cp2k_sp."""
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    all_jobs = []
    for i, atoms in enumerate(frames):
        # Use a val-specific name that can never collide with sp_*_r{N}_* jobs
        sys_type = atoms.info.get("system_type", "unknown")
        name     = f"{JOB_PREFIX}_{sys_type}_{i:04d}"

        # Temporarily tag so write_cp2k_sp naming is consistent
        atoms.info.setdefault("system_type", sys_type)

        inp = write_cp2k_sp(atoms, name, str(VAL_DIR))
        all_jobs.append((name, inp))
        print(f"  [+] {name}.inp")

    print(f"\n[+] Wrote {len(all_jobs)} inputs → {VAL_DIR}/")

    # Two scripts: all jobs and only missing (incomplete .out)
    missing_jobs = [
        (name, inp) for name, inp in all_jobs
        if not _cp2k_output_is_complete(VAL_DIR / f"{name}.out")
    ]

    _write_submission_script(
        VAL_DIR / "submit_all.sh",
        all_jobs,
        str(VAL_DIR),
        label="ALL validation jobs",
        n_total=len(all_jobs),
        n_skipped=0,
    )
    _write_submission_script(
        VAL_DIR / "submit_missing.sh",
        missing_jobs,
        str(VAL_DIR),
        label="missing validation jobs",
        n_total=len(all_jobs),
        n_skipped=len(all_jobs) - len(missing_jobs),
    )

    print(f"[+] submit_all.sh     — {len(all_jobs)} jobs")
    print(f"[+] submit_missing.sh — {len(missing_jobs)} jobs need CP2K")

    if not missing_jobs:
        print("[✓] All outputs already present — run --parse directly.")

    return all_jobs, missing_jobs


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Parse outputs
#
#  We cannot call parse_cp2k_sp_results directly because it uses the
#  global ROUND to reconstruct job names as sp_{sys}_r{ROUND}_{i:04d}.
#  Our val jobs are named val_{sys}_{i:04d} so we reimplement the
#  name reconstruction here, reusing all the parsing internals.
# ─────────────────────────────────────────────────────────────────────────────

def step_parse(frames):
    """
    Parse VAL/*.out files and write fps_validate_framesV2.xyz.
    Reuses parse_stress_from_out and the same key names (REF_energy,
    REF_forces, REF_stress) as the production pipeline.
    """
    parsed  = []
    failed  = []
    n_total = len(frames)

    for i, atoms in enumerate(frames):
        sys_type = atoms.info.get("system_type", "unknown")
        name     = f"{JOB_PREFIX}_{sys_type}_{i:04d}"
        outfile  = VAL_DIR / f"{name}.out"

        print(f"\n[->] Parsing {name}.out ...")

        # ── completeness check ───────────────────────────────────────────
        if not outfile.exists():
            print(f"  [!] Missing: {outfile}")
            failed.append((i, name, "missing"))
            continue

        content = outfile.read_text()

        if "SCF run NOT converged" in content:
            print(f"  [!] SCF not converged: {name}")
            failed.append((i, name, "scf_failed"))
            continue

        # ── energy ───────────────────────────────────────────────────────
        energy_match = re.search(
            r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-\d.]+)",
            content,
        )
        if not energy_match:
            print(f"  [!] Energy not found: {name}")
            failed.append((i, name, "no_energy"))
            continue

        energy_eV = float(energy_match.group(1)) * HA_TO_EV
        print(f"  [E] {energy_eV:.6f} eV")

        # ── forces ───────────────────────────────────────────────────────
        force_block = re.search(
            r"ATOMIC FORCES in \[a\.u\.\](.*?)SUM OF ATOMIC FORCES",
            content,
            re.DOTALL,
        )
        forces_ok = False
        forces    = []

        if force_block:
            for line in force_block.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) == 6:
                    fx = float(parts[3]) * HA_BOHR_TO_EV_ANG
                    fy = float(parts[4]) * HA_BOHR_TO_EV_ANG
                    fz = float(parts[5]) * HA_BOHR_TO_EV_ANG
                    forces.append([fx, fy, fz])

            if len(forces) == len(atoms):
                forces_ok = True
                f_max = np.max(np.linalg.norm(forces, axis=1))
                print(f"  [F] {len(forces)} forces parsed, max = {f_max:.3f} eV/Å")
            else:
                print(f"  [!] Force count mismatch: got {len(forces)}, "
                      f"expected {len(atoms)}")

        # ── stress ───────────────────────────────────────────────────────
        stress = parse_stress_from_out(content)
        if stress is not None:
            voigt = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
            print(f"  [σ] Stress parsed (Voigt): "
                  f"{' '.join(f'{v:.4f}' for v in voigt)} eV/Å³")
        else:
            print(f"  [~] No stress tensor found")

        # ── assemble result frame ─────────────────────────────────────────
        result             = atoms.copy()
        result.calc        = None
        result.info["REF_energy"]   = energy_eV
        result.info["source"]       = name
        result.info["system_type"]  = sys_type
        result.info["validation"]   = True     # explicit flag — never train on these

        if forces_ok:
            result.arrays["REF_forces"] = np.array(forces)

        if stress is not None:
            result.info["REF_stress"] = voigt

        parsed.append(result)

    # ── write output ─────────────────────────────────────────────────────────
    if parsed:
        # Safety: refuse to write if OUTPUT_XYZ is the training pool
        assert OUTPUT_XYZ not in ("master_train_pool.xyz", "training_clean.xyz"), \
            "OUTPUT_XYZ must not point to a training file!"

        write(OUTPUT_XYZ, parsed, format="extxyz")
        print(f"\n[✓] Wrote {len(parsed)} frames → {OUTPUT_XYZ}")
    else:
        print("\n[!] No frames parsed successfully — check VAL/*.out files")

    # ── failure report ────────────────────────────────────────────────────────
    if failed:
        print(f"\n[!] {len(failed)} frames failed:")
        for idx, name, reason in failed:
            print(f"    frame {idx:04d}  {name}  ({reason})")

        fail_log = VAL_DIR / "failed_jobs.txt"
        fail_log.write_text(
            "\n".join(f"{idx}\t{name}\t{reason}" for idx, name, reason in failed)
        )
        print(f"    → {fail_log}")

    print(f"\n    Total: {n_total}  parsed: {len(parsed)}  failed: {len(failed)}")
    print(f"\n[!] {OUTPUT_XYZ} is a VALIDATION FILE.")
    print(f"    Do NOT add it to master_train_pool.xyz or training_clean.xyz.")

    return parsed, failed


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true",
                       help="Write CP2K inputs and submit scripts only")
    group.add_argument("--parse", action="store_true",
                       help="Parse completed CP2K outputs → OUTPUT_XYZ")
    group.add_argument("--all",   action="store_true",
                       help="Write inputs, run CP2K locally, then parse")
    args = parser.parse_args()

    # ── safety: never overwrite a completed validation set silently ───────
    if (args.parse or args.all) and Path(OUTPUT_XYZ).exists():
        ans = input(f"[?] {OUTPUT_XYZ} already exists. Overwrite? [y/N]: ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            return

    print(f"[->] Reading: {INPUT_XYZ}")
    frames = read(INPUT_XYZ, index=":")
    print(f"[+]  Frames: {len(frames)}")

    if args.write or args.all:
        print("\n── STEP 1: Writing inputs ──────────────────────────────────")
        all_jobs, missing_jobs = step_write(frames)

        if args.all and missing_jobs:
            print("\n── STEP 2: Running CP2K locally ────────────────────────────")
            import subprocess
            submit = VAL_DIR / "submit_missing.sh"
            subprocess.run(["bash", str(submit)], check=True)

    if args.parse or args.all:
        print("\n── STEP 3: Parsing outputs ─────────────────────────────────")
        step_parse(frames)


if __name__ == "__main__":
    main()