"""
pool_coverage.py

Compares your candidate pool (NEB/GeoOpt frames) against your master
training pool to show what percentage of available chemistry you have
DFT data for. Use this to decide whether to rerun neb_geo_run.py.

Usage: python pool_coverage.py
"""

import numpy as np
import hashlib
from pathlib import Path
from collections import defaultdict
from ase.io import read

# ── Config ──────────────────────────────────────────────────────────────────
CANDIDATE_DIR   = "geo_opt_results/al_candidates"       # where mace_neb_*.extxyz and mace_geoopt_*.extxyz live
POOL_FILE       = "master_train_pool.xyz"
TRAINING_FILE   = "training_clean.xyz"  # post-cleaning file used for training
# ─────────────────────────────────────────────────────────────────────────────

def get_atoms_hash(atoms):
    pos_data = np.round(atoms.get_positions(), 4).tobytes()
    nuc_data = atoms.get_atomic_numbers().tobytes()
    return hashlib.md5(pos_data + nuc_data).hexdigest()

def load_frames_with_source(directory):
    """Load all candidate frames, tagging each with its source file type."""
    frames = []
    for path in sorted(Path(directory).glob("*.extxyz")):
        name = path.stem  # e.g. mace_neb_Close0.25PtOH2
        if "neb" in name:
            frame_type = "NEB"
        elif "geoopt" in name:
            frame_type = "GeoOpt"
        elif "plumed" in name:
            frame_type = "PLUMED"
        else:
            frame_type = "Other"

        try:
            batch = read(str(path), ":")
            for atoms in batch:
                atoms.info["_source_file"] = path.name
                atoms.info["_frame_type"]  = frame_type
                if "system_type" not in atoms.info:
                    # Extract system type from filename
                    # mace_neb_Close0.25PtOH2 → Close0.25PtOH2
                    atoms.info["system_type"] = name.replace(
                        "mace_neb_", ""
                    ).replace("mace_geoopt_", "").replace("mace_plumed_", "")
            frames.extend(batch)
        except Exception as e:
            print(f"  [!] Could not load {path.name}: {e}")

    return frames

def main():
    print("\n" + "="*65)
    print("  Candidate Pool Coverage Report")
    print("="*65)

    # ── Load candidate pool ──────────────────────────────────────────
    print(f"\n[→] Loading candidates from {CANDIDATE_DIR}/...")
    candidates = load_frames_with_source(CANDIDATE_DIR)
    if not candidates:
        print(f"  [!] No candidates found in {CANDIDATE_DIR}/")
        return
    print(f"    Total candidate frames: {len(candidates)}")

    # ── Load master pool ─────────────────────────────────────────────
    pool_hashes = set()
    if Path(POOL_FILE).exists():
        print(f"[→] Loading master pool: {POOL_FILE}...")
        pool_frames = read(POOL_FILE, ":")
        pool_hashes = {get_atoms_hash(a) for a in pool_frames}
        print(f"    Frames in master pool: {len(pool_frames)}")
    else:
        print(f"  [!] {POOL_FILE} not found")

    # ── Load training set ────────────────────────────────────────────
    train_hashes = set()
    if Path(TRAINING_FILE).exists():
        print(f"[→] Loading training set: {TRAINING_FILE}...")
        train_frames = read(TRAINING_FILE, ":")
        train_hashes = {get_atoms_hash(a) for a in train_frames}
        print(f"    Frames in training set (post-cleaning): {len(train_frames)}")
    else:
        print(f"  [!] {TRAINING_FILE} not found — skipping training set stats")

    # ── Build per-system-type breakdown ─────────────────────────────
    # candidate stats
    sys_total     = defaultdict(int)
    sys_in_pool   = defaultdict(int)
    sys_in_train  = defaultdict(int)
    sys_type_map  = defaultdict(lambda: defaultdict(int))  # stype → frame_type → count

    for atoms in candidates:
        stype      = atoms.info.get("system_type", "unknown")
        ftype      = atoms.info.get("_frame_type", "?")
        h          = get_atoms_hash(atoms)

        sys_total[stype]            += 1
        sys_type_map[stype][ftype]  += 1

        if h in pool_hashes:
            sys_in_pool[stype] += 1
        if h in train_hashes:
            sys_in_train[stype] += 1
    
    # ── Overall summary ──────────────────────────────────────────────
    total      = len(candidates)
    in_pool    = sum(1 for a in candidates if get_atoms_hash(a) in pool_hashes)
    in_train   = sum(1 for a in candidates if get_atoms_hash(a) in train_hashes)
    remaining  = total - in_pool

    print(f"\n{'='*65}")
    print(f"  Overall Coverage")
    print(f"{'='*65}")
    print(f"  Total candidate frames    : {total:>6}")
    print(f"  In master pool (computed) : {in_pool:>6}  ({100*in_pool/total:.1f}%)")
    print(f"  In training set (clean)   : {in_train:>6}  ({100*in_train/total:.1f}%)")
    print(f"  Not yet computed          : {remaining:>6}  ({100*remaining/total:.1f}%)")

    # ── Recommendation ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Recommendation")
    print(f"{'='*65}")
    pct_computed = 100 * in_pool / total
    if pct_computed < 40:
        print(f"  ✅ Pool is {pct_computed:.0f}% computed — plenty of new frames available.")
        print(f"     Continue active learning without rerunning neb_geo_run.py.")
    elif pct_computed < 70:
        print(f"  ⚠️  Pool is {pct_computed:.0f}% computed — getting sparse.")
        print(f"     Consider rerunning neb_geo_run.py with the fine-tuned model")
        print(f"     after the next training round to refresh candidates.")
    else:
        print(f"  🔴 Pool is {pct_computed:.0f}% computed — candidate pool nearly exhausted.")
        print(f"     Rerun neb_geo_run.py with your latest fine-tuned model before")
        print(f"     starting the next active learning round.")

    # ── Per-system breakdown ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Per-System Coverage")
    print(f"{'='*65}")
    print(f"  {'System':<25} {'Total':>6} {'Pool':>6} {'Train':>6} "
          f"{'Pool%':>7} {'Train%':>7} {'Remaining':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*10}")

    # Sort by pool coverage ascending — least covered systems first
    for stype in sorted(sys_total, key=lambda s: sys_in_pool[s]/sys_total[s]):
        n_total   = sys_total[stype]
        n_pool    = sys_in_pool[stype]
        n_train   = sys_in_train[stype]
        n_remain  = n_total - n_pool
        pct_pool  = 100 * n_pool  / n_total
        pct_train = 100 * n_train / n_total

        # Flag systems with zero training data
        flag = " ← NO TRAINING DATA" if n_train == 0 else ""
        flag = " ← POOL EXHAUSTED"   if n_remain == 0 and n_total > 0 else flag

        print(f"  {stype:<25} {n_total:>6} {n_pool:>6} {n_train:>6} "
              f"{pct_pool:>6.0f}% {pct_train:>6.0f}% {n_remain:>10}  {flag}")

    # ── Frame type breakdown ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Frame Type Breakdown")
    print(f"{'='*65}")
    type_totals = defaultdict(int)
    type_pool   = defaultdict(int)
    for atoms in candidates:
        ft = atoms.info.get("_frame_type", "?")
        h  = get_atoms_hash(atoms)
        type_totals[ft] += 1
        if h in pool_hashes:
            type_pool[ft] += 1

    print(f"  {'Type':<12} {'Total':>6} {'In Pool':>8} {'Coverage':>10} {'Remaining':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
    for ft, n in sorted(type_totals.items()):
        np_ = type_pool[ft]
        print(f"  {ft:<12} {n:>6} {np_:>8} {100*np_/n:>9.0f}% {n-np_:>10}")

    print(f"\n{'='*65}\n")

if __name__ == "__main__":
    main()