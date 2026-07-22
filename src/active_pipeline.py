#!/usr/bin/env python3
"""
active_pipeline.py
==================
Round-based active learning for MACE fine-tuning.
 
WORKFLOW
--------
  Step 0  mace_geo_opt_base.py  runs GeoOpt + NEB and writes tagged .extxyz
          files into geo_opt_results/al_candidates/:
              mace_geoopt_{name}.extxyz   — optimised initial/final endpoints
              mace_neb_{name}.extxyz      — all NEB images
 
  Step 1  THIS SCRIPT reads those files, scores frames by MACE force magnitude,
          and picks a diverse subset with Farthest-Point Sampling (FPS).
          Optionally appends FPS-sampled frames from dissolved-system MD
          trajectories (bulk water + Nafion).
 
  Step 2  Writes CP2K single-point input files (one per selected frame).
 
  Step 3  (After CP2K runs)  Parse CP2K outputs, convert units, attach
          energy + forces as REF_energy / REF_forces, and merge into a
          growing pool XYZ file that MACE-Train accepts directly.
 
USAGE
-----
  # After mace_geo_opt_base.py has finished:
  python active_pipeline.py
 
  # After CP2K jobs have finished:
  python active_pipeline.py --parse
 
  # Re-try only the frames listed in failed_jobs.txt (after fixing/rerunning them):
  python active_pipeline.py --reparse
  
NOTES
-----
This script can also be used to generate the E0s values for fine tunning (see line 90)
"""

import os
import re
import sys
import json
from ase.config import cfg
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.units import Hartree, Bohr
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from MACE_CP2K_pipeline.src.patches import apply_dftd3_cell_patch
apply_dftd3_cell_patch()

# ============================================================
# Configuration
# ============================================================

ROUND       = 1          # Increment each iteration
_FOUNDATION_MODEL = os.environ.get("MACE_FOUNDATION_MODEL", "mace-mp-0b3-medium-float32.model")
MODEL_PATH = (f"mace_V{ROUND-1}_active_learning_stagetwo.model" if ROUND > 3
              else _FOUNDATION_MODEL)

# Where neb_geo_run.py wrote its AL candidate files
AL_INPUT_DIR  = "geo_opt_results/al_candidates"

# How many frames to send to CP2K per round (total across all systems)
N_SELECT_TOTAL = 100
MAX_ATOMS = 350.0 # This is to prevent the data set becoming too large and wasting gpu
REUSE_EXISTING_CP2K = True # If true, skip writing inputs for frames that already have valid CP2K outputs (useful for iterative rounds)
EXCLUDE_SYSTEM_KEYWORDS = [] 

def apply_round(n):
    """
    Update all round-dependent globals to match round N.
    Called once at startup after argument parsing.
    """
    global ROUND, MODEL_PATH, CP2K_DIR, FAILED_LOG
    ROUND      = n
    MODEL_PATH = (f"mace_V{ROUND-1}_active_learning_stagetwo.model" if ROUND > 3
               else "mace-mp-0b3-medium-float32.model")
    CP2K_DIR   = f"cp2k_sp_round{n}"
    FAILED_LOG = f"cp2k_sp_round{n}/failed_jobs.txt"
    print(f"[→] Round {n}  |  Model: {MODEL_PATH}  |  CP2K dir: {CP2K_DIR}")



# --- Pathological-geometry triage before CP2K ---
# If a candidate's *initial* MACE force exceeds this, it's far more likely to
# be an overlap/clash artifact (e.g. raw FPS pick from a foreign dataset's
# cell) than a genuinely interesting AL frame. We run a short, CAPPED
# relaxation to remove the overlap -- not a full optimisation, since fully
# converging would erase the very off-equilibrium character that makes a
# frame worth sending to CP2K in the first place.
GEOOPT_TRIGGER       = True   # Turns this feature on or off
GEOOPT_TRIGGER_FORCE = 20.0   # eV/Å -- well above FORCE_THRESHOLD; this flags "broken", not "uncertain"
GEOOPT_MAX_STEPS     = 30     # hard cap -- keep this cheap and avoid fully annealing the frame
GEOOPT_FMAX_TARGET   = 2.0    # eV/Å -- loose target: "no longer exploding", not "converged minimum"
APPLY_D3             = True   # Whether to include D3 in all calculations (MACE + D3) for this round. If False, only MACE is used.


EXTERNAL_DATASETS = False  # add this to your config flags at the top

# Additional data sets to parse for training
EXTERNAL_SOURCES = {
    #"mptrj_pt": {
    #    "path": "training_data/mptrj-gga-ggapu/master_ranked_MPtrj_structures.extxyz",
    #    "n_samples": 30,   # only 35 frames total — take most of them
    #},
    #"oc25_pt": {
    #    "path": "hugface_data/train/master_ranked_OC25_structures.extxyz",
    #    "n_samples": 90,   # 93 frames total — sample ~30
    #},
}

REICO_SAMPLEING = True
# --- REICO (random imaginary-chemical box) sampling config ---
REICO_NUM            = 100     # number of random boxes generated per round
REICO_MIN_ATOMS      = 20
REICO_MAX_ATOMS      = 50
REICO_VOL_PER_ATOM   = 16.0   # Å³/atom -- rough condensed-phase packing density,
                               # box edge is derived from this + n_atoms so boxes
                               # stay dense rather than dilute-gas-like
REICO_MIN_DIST_SCALE = 0.6    # scales (covalent_radius_a + covalent_radius_b) to
                               # get a per-pair minimum distance, instead of one
                               # global cutoff that's wrong for both H-H and Pt-Pt

# Output paths
CP2K_DIR   = f"cp2k_sp_round{ROUND}"
POOL_FILE  = "master_train_pool.xyz"
CP2K_TIMEOUT = "3h"  # Per-job timeout for CP2K runs (adjust as needed)
FAILED_LOG = f"{CP2K_DIR}/failed_jobs.txt"   # written by your timeout wrapper

# E0s Json
E0_JSON = "E0s.json"
E0_DIR  = f"cp2k_e0_round{ROUND}"
E0_CELL_SIZE = 20.0  # 20x20x20 Angstrom box
# Mapping for atomic numbers 
Z_MAP = {"H": 1, "Li": 3, "C": 6, "O": 8, "F": 9, "P": 15, "S": 16, "Pt": 78}

# ============================================================
# MACE re-scoring helper
# ============================================================
def rescore_with_mace(frames, model_path, device="cuda", dtype="float32"):
    """
    Attach a fresh MACE calculator to every frame and compute forces.
    Needed when frames were loaded from disk without a live calculator.
    """
    try:
        from mace.calculators import MACECalculator
    except ImportError:
        print("[!] mace not importable — skipping re-score, using stored forces.")
        return frames

    calc_mace = MACECalculator(model_paths=model_path, device=device, default_dtype=dtype)
    if APPLY_D3:
        print(f"[→] Re-scoring with MACE + D3 (device={device}, dtype={dtype})...")
        calc_DFT = TorchDFTD3Calculator(
            device=device,
            damping="bj",
            xc=cfg.get("dispersion_xc", "pbe"),
            cutoff=cfg.get("dispersion_cutoff", 40.0),
        )
        calc = SumCalculator([calc_mace, calc_DFT])
    else:
        print(f"[→] Re-scoring with MACE only (device={device}, dtype={dtype})...")
        calc = calc_mace
    for atoms in frames:
        atoms.calc = calc
        try:
            atoms.get_potential_energy()   # triggers force evaluation
        except Exception as e:
            print(f"    [!] Re-score failed for one frame: {e}")
    return frames

# ============================================================
# MD trajectory sampler — FPS diversity, NOT force-scored
# ============================================================
 
def fps_sample_md_trajectory(frames, n_select, system_name):
    """
    Pick n_select structurally diverse frames from an MD trajectory using FPS.
 
    WHY NOT force-score these?
    MD trajectories are generated at finite temperature — every frame has
    large, meaningful forces by design. Force-scoring would just pick the
    most-distorted snapshots (high-T fluctuations), which are not the most
    informative for training. Instead, FPS on atomic positions gives you
    n_select frames spread evenly across the sampled configuration space.
 
    Strategy
    --------
    1. Stride the trajectory first (max 500 frames) so FPS is fast.
    2. Run FPS on flattened, normalised position vectors.
    3. Tag each frame with system_type and source before returning.
    """
    from sklearn.preprocessing import normalize
 
    # Stride to keep FPS tractable for long trajectories
    MAX_POOL = 500
    if len(frames) > MAX_POOL:
        stride = len(frames) // MAX_POOL
        frames = frames[::stride]
    print(f"  [{system_name}] Pool after stride: {len(frames)} frames")
 
    if len(frames) <= n_select:
        chosen = list(range(len(frames)))
    else:
        # Build feature matrix — use first FEATURE_DIM position coords
        FEATURE_DIM = 300
        features = []
        for atoms in frames:
            pos = atoms.get_positions().flatten()
            if len(pos) >= FEATURE_DIM:
                features.append(pos[:FEATURE_DIM])
            else:
                features.append(np.pad(pos, (0, FEATURE_DIM - len(pos))))
 
        features  = normalize(np.array(features))
        chosen    = [0]
        min_dists = np.full(len(frames), np.inf)
 
        for _ in range(n_select - 1):
            dists     = np.linalg.norm(features - features[chosen[-1]], axis=1)
            min_dists = np.minimum(min_dists, dists)
            chosen.append(int(np.argmax(min_dists)))
 
    selected = []
    for idx in chosen:
        at = frames[idx].copy()
        at.info["system_type"] = system_name
        at.info["source"]      = "md_trajectory"
        at.calc = None
        selected.append(at)
 
    return selected

# ============================================================
# Step 1 — Load candidate frames produced by neb_geo_run.py
# ============================================================

def _is_excluded(system_type: str) -> bool:
    """Return True if system_type matches any exclusion keyword (case-insensitive)."""
    s = system_type.lower()
    return any(kw.lower() in s for kw in EXCLUDE_SYSTEM_KEYWORDS)

def load_candidates(al_input_dir):
    """
    Read every .extxyz in al_input_dir.
    Each frame must already carry  atoms.info["system_type"].
    Returns a flat list of ASE Atoms objects.
    """
    input_dir = Path(al_input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"AL input directory not found: {al_input_dir}\n"
            "Run neb_geo_run.py first."
        )

    all_frames = []
    xyz_files  = sorted(input_dir.glob("*.extxyz"))

    if not xyz_files:
        raise FileNotFoundError(f"No .extxyz files found in {al_input_dir}")

    for xyz_path in xyz_files:
        frames = read(str(xyz_path), index=":")
        filtered = []
        for atoms in frames:
            if "system_type" not in atoms.info:
                atoms.info["system_type"] = xyz_path.stem
            if _is_excluded(atoms.info["system_type"]):
                continue   # drop it early
            filtered.append(atoms)

        skipped = len(frames) - len(filtered)
        if skipped:
            print(f"  Excluded {skipped:3d} frames   ←  {xyz_path.name} (keyword match)")
        all_frames.extend(filtered)
        print(f"  Loaded {len(filtered):4d} frames  ←  {xyz_path.name}")

    print(f"\nTotal candidate frames: {len(all_frames)}")
    return all_frames

# ============================================================
# Step 2 — Select uncertain frames: force scoring + FPS
# ============================================================

def select_uncertain_frames(frames, n_select, force_threshold=None):
    """
    Score by MACE max-force, then apply FPS for geometric diversity.
    Returns a list of selected Atoms objects.
    """
    from sklearn.preprocessing import normalize

    # --- Score ---
    scores = []
    print("\nScoring frames by MACE max force...")
    for atoms in frames:
        if atoms.calc is not None:
            try:
                f = atoms.get_forces()
                scores.append(float(np.max(np.linalg.norm(f, axis=1))))
            except Exception:
                scores.append(0.0)
        else:
            scores.append(0.0)

    scores = np.array(scores)

    # Pre-filter: prefer frames above threshold, fall back to top-N if too few
    print(f"\nPre-filtering candidates with force threshold: {force_threshold} eV/Å")
    if force_threshold is not None:
        candidates = np.where(scores > force_threshold)[0]
        if len(candidates) < n_select:
            candidates = np.argsort(scores)[::-1][:n_select * 2]
    else:
        candidates = np.argsort(scores)[::-1][:n_select * 2]

    if len(candidates) <= n_select:
        return [frames[i] for i in candidates]

    # --- FPS for diversity ---
    FEATURE_DIM = 300
    features = []
    print(f"\nApplying FPS to {len(candidates)} candidates (force threshold: {force_threshold})...")
    for idx in candidates:
        pos = frames[idx].get_positions().flatten()
        if len(pos) >= FEATURE_DIM:
            features.append(pos[:FEATURE_DIM])
        else:
            features.append(np.pad(pos, (0, FEATURE_DIM - len(pos))))

    features   = normalize(np.array(features))
    selected   = [0]
    min_dists  = np.full(len(candidates), np.inf)

    for _ in range(n_select - 1):
        dists     = np.linalg.norm(features - features[selected[-1]], axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))
    print(f"  Selected candidate indices: {[candidates[i] for i in selected]}")
    return [frames[candidates[i]] for i in selected]

# ============================================================
# Step 3 — CP2K single-point input generation
# ============================================================
LIBDIR = os.environ.get("CP2K_LIBDIR")
if not LIBDIR:
    raise ValueError("CP2K_LIBDIR environment variable is not set! Did you source config.local.sh?")

CP2K_TEMPLATE = """\
&GLOBAL
  PROJECT_NAME {name}
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD QS
  &DFT
    BASIS_SET_FILE_NAME {LIBDIR}/BASIS_MOLOPT
    POTENTIAL_FILE_NAME {LIBDIR}/GTH_POTENTIALS
    &MGRID
      CUTOFF 500
      REL_CUTOFF 50
      NGRIDS 5
    &END MGRID
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      MAX_SCF 150
      EPS_SCF 1.0E-6
      ADDED_MOS 500
      &SMEAR ON
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE [K] 1000
      &END SMEAR
      &MIXING
        METHOD BROYDEN_MIXING
        ALPHA 0.1
        BETA 1.0
        NBROYDEN 12
      &END MIXING
      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
      &vdW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          REFERENCE_FUNCTIONAL PBE
          PARAMETER_FILE_NAME {LIBDIR}/dftd3.dat
        &END PAIR_POTENTIAL
      &END vdW_POTENTIAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC {a:.6f} {b:.6f} {c:.6f}
      PERIODIC XYZ
    &END CELL
    &COORD
{coords}
    &END COORD
{kinds}
  &END SUBSYS
  &PRINT
    &STRESS_TENSOR
    &END STRESS_TENSOR 
    &FORCES
    &END FORCES
  &END PRINT
  STRESS_TENSOR ANALYTICAL 
&END FORCE_EVAL
"""

KIND_TEMPLATE = """\
    &KIND {symbol}
      BASIS_SET {basis}
      POTENTIAL {potential}
    &END KIND"""

KIND_PARAMS = {
    "H":  ("DZVP-MOLOPT-SR-GTH-q1",  "GTH-PBE-q1"),
    "C":  ("DZVP-MOLOPT-SR-GTH-q4",  "GTH-PBE-q4"),
    "O":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "F":  ("DZVP-MOLOPT-SR-GTH-q7",  "GTH-PBE-q7"),
    "S":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "Pt": ("DZVP-MOLOPT-SR-GTH-q18", "GTH-PBE-q18"),
    # Add elements P, Li, B, N, 
}

if KIND_PARAMS == {    "Element":  ("DZVP-MOLOPT-SR-GTH",  "GTH-PBE")}:
    print(f" You did not define your Kinds in the active pipline file")


# Default cell dimensions per system type — used when a structure has no cell
# (e.g. isolated molecules read from .cif without periodic boundary info).
# Keys must match the system_type tag set by mace_geo_opt_base.py exactly
# (case-insensitive comparison is used below).
# takes a keyword that could be found in a structure name and gives it the following cell
 
DEFAULT_CELLS = {
    # Pt slab systems  (a, b, c) in Angstrom
    "default":             (11.099, 9.612,  33.000),
    # Nafion + Pt slab
    "naf_naf":             (22.198, 19.224, 29.790),
    "pt_nafion":           (22.198, 19.224, 29.790),
    # Bulk dissolved systems
    "bulk_nafion":         (11.099, 9.612,  21.220),
    "bulk_water_pt":       (11.099, 9.612,  21.220),
    # Dissolved oxide/hydroxide in Nafion
    "dissolvedoh_nafion":  (11.099, 9.612,  33.000),
    "dissolvedo2_nafion":  (11.099, 9.612,  33.000),
    "dissolvedo_nafion":   (11.099, 9.612,  33.000),
}


def _resolve_cell(atoms, name):
    """
    Return (a, b, c) cell lengths for a CP2K input.
    Uses the actual cell if non-zero, otherwise falls back to DEFAULT_CELLS
    keyed on atoms.info["system_type"].
    """
    cell = atoms.get_cell()
    a, b, c = cell[0, 0], cell[1, 1], cell[2, 2]

    if a != 0.0 or b != 0.0 or c != 0.0:
        return a, b, c   # cell already set — use it

    sys_type = atoms.info.get("system_type", "").lower().replace("-", "_")
    a, b, c  = DEFAULT_CELLS.get(sys_type, DEFAULT_CELLS["default"])
    print(f"    [!] Zero cell for {name} (system_type={sys_type!r}) "
          f"— using default {a:.3f} x {b:.3f} x {c:.3f} A"
          f" You can modify this in the python file at line 385")
    return a, b, c


def write_cp2k_sp(atoms, name, outdir):
    """Write one CP2K single-point input file."""
    os.makedirs(outdir, exist_ok=True)

    a, b, c = _resolve_cell(atoms, name)
    print(f"  Writing CP2K input for {name}: "
          f"{len(atoms)} atoms  cell = {a:.6f} {b:.6f} {c:.6f}")
        

    coords = ""
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        coords += f"      {sym:<4s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}\n"

    symbols_present = sorted(set(atoms.get_chemical_symbols()))
    kinds = "\n".join(
        KIND_TEMPLATE.format(
            symbol=s,
            basis=KIND_PARAMS[s][0],
            potential=KIND_PARAMS[s][1]
        )
        for s in symbols_present
        if s in KIND_PARAMS
    )

    missing = [s for s in symbols_present if s not in KIND_PARAMS]
    if missing:
        print(f"  [!] No KIND_PARAMS for: {missing}  — they will be absent from {name}.inp")

    inp = CP2K_TEMPLATE.format(
        name=name, a=a, b=b, c=c,
        coords=coords.rstrip(), kinds=kinds,
        LIBDIR=LIBDIR
    )

    inp_file = Path(outdir) / f"{name}.inp"
    inp_file.write_text(inp)
    return str(inp_file)

import hashlib
HASH_PRECISION = 4
def get_atoms_hash(atoms):
    """
    MD5 fingerprint of an Atoms object based on positions + atomic numbers.
    Positions are rounded to 4 decimal places (0.1 mA precision) so that
    tiny floating-point noise does not produce spurious mismatches.
    Used to detect duplicate geometries across rounds so CP2K is not rerun
    on structures that were already calculated.
    """
    pos_data = np.round(atoms.get_positions(), HASH_PRECISION).tobytes()
    nuc_data = atoms.get_atomic_numbers().tobytes()
    return hashlib.md5(pos_data + nuc_data).hexdigest()

GLOBAL_GEOM_INDEX = "geometry_index_global.json"

def _load_geometry_index(cp2k_dir):
    """
    Load geometry hash index. Handles both formats:
    - format: {hash: {"name": "job_name", "cp2k_dir": "cp2k_sp_round1"}}
    """
    import json

    # Check for global index first (new multi-round aware index)
    global_path = Path(GLOBAL_GEOM_INDEX)
    round_path  = Path(cp2k_dir) / "geometry_index.json"

    # Load whichever exists, preferring global
    index_path = global_path if global_path.exists() else round_path
    if not index_path.exists():
        return {}

    with open(index_path) as f:
        raw = json.load(f)

    combined = {}
    for h, entry in raw.items():
        # --- Normalise to new dict format ---
        if isinstance(entry, str):
            # Old format — job name only, assume it lives in cp2k_dir
            normalised = {"name": entry, "cp2k_dir": str(cp2k_dir)}
        elif isinstance(entry, dict):
            # New format — already has name + cp2k_dir
            normalised = entry
        else:
            continue  # corrupt entry, skip

        # Validate the .out file is actually complete
        out = Path(normalised["cp2k_dir"]) / f"{normalised['name']}.out"
        if _cp2k_output_is_complete(out):
            combined[h] = normalised

    return combined

def _save_geometry_index(cp2k_dir, index):
    """Save index to global file in new dict format."""
    import json
    # Always write to global index — covers all rounds
    with open(GLOBAL_GEOM_INDEX, "w") as f:
        json.dump(index, f, indent=2)

def _cp2k_output_is_complete(outfile):
    """
    Return True if a CP2K .out file exists, is non-empty, contains a parsed
    energy line, and did NOT hit an SCF convergence failure.
    This is the completeness check used by REUSE_EXISTING_CP2K.
    """
    p = Path(outfile)
    if not p.exists() or p.stat().st_size == 0:
        return False
    content = p.read_text()
    if "SCF run NOT converged" in content:
        return False
    if not re.search(
        r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+[-\d.]+",
        content
    ):
        return False
    return True

def _write_submission_script(path, jobs, cp2k_dir, label, n_total, n_skipped, append=False):
    """Write a bash submission script for the given list of (name, inp) jobs."""
    mode = "a" if append and path.exists() else "w"
    skip_comment = (
        f"# {n_skipped} of {n_total} frames skipped (completed .out already present)\n"
        if n_skipped else ""
    )
    header = f"""\
#!/bin/bash
set -uo pipefail
# CP2K single-point calculations - Round {ROUND} ({label})
{skip_comment}

timestamp=$(date +%Y%m%d_%H%M)
start_time=$(date +%s)
counter=0
export OMP_NUM_THREADS=6
total={len(jobs)}

# --- memory guard settings ---
MEM_LIMIT_KB=$(( 62 * 1024 * 1024 * 85 / 100 ))   # 85% of 62GB, in KB
MEM_CHECK_INTERVAL=2                               # seconds between checks
FAILED_LOG={cp2k_dir}/failed_jobs.txt
TIMES_LOG={cp2k_dir}/job_times.log

# Hard backstop in case the polling watchdog misses a fast spike.
# Virtual memory ulimit in KB; set slightly above MEM_LIMIT_KB so the
# watchdog (softer, faster to react) is normally the one that fires.
ulimit -v $(( MEM_LIMIT_KB * 110 / 100 ))

print_eta() {{
    if (( counter > 0 && elapsed > 0 )); then
        avg=$(( elapsed / counter ))
        eta=$(( avg * (total - counter) ))
        echo "  ETA: ~$(( eta / 60 ))m remaining"
    fi
}}
mem_watchdog() {{
    local target_pid=$1
    local job_name=$2
    while kill -0 "$target_pid" 2>/dev/null; do
        pids="$target_pid $(pgrep -P "$target_pid" 2>/dev/null)"
        rss=$(ps -o rss= -p $pids 2>/dev/null | awk '{{sum+=$1}} END {{print sum+0}}')
        if (( rss > 0 && rss > MEM_LIMIT_KB )); then
            echo "job killed because of memory: {{$job_name}} (RSS=${{rss}}KB > limit=${{MEM_LIMIT_KB}}KB)"
            kill -TERM "$target_pid" 2>/dev/null
            sleep 2
            kill -KILL "$target_pid" 2>/dev/null
            echo "MEMKILL: $job_name" >> "$FAILED_LOG"
            break
        fi
        sleep "$MEM_CHECK_INTERVAL"
    done
}}

cleanup() {{
    echo "Interrupted at job $counter/$total"
    rm -f *.wfn *.wfn.bak-1
    rm -f {cp2k_dir}/*.wfn {cp2k_dir}/*.wfn.bak-1
    exit 1
}}
trap cleanup INT TERM
"""
    
    job_blocks = []
    for name, inp in jobs:
        block = f"""\
        counter=$((counter + 1))
        job_start=$(date +%s)
        echo "[$counter/$total] {name} — started $(date +%H:%M)"

        timeout {CP2K_TIMEOUT} cp2k.ssmp -i {inp} -o {cp2k_dir}/{name}.out &
        cp2k_pid=$!
        mem_watchdog "$cp2k_pid" "{name}" &
        watchdog_pid=$!

        wait "$cp2k_pid"
        cp2k_status=$?
        kill "$watchdog_pid" 2>/dev/null
        wait "$watchdog_pid" 2>/dev/null

        if grep -q "MEMKILL: {name}" "$FAILED_LOG" 2>/dev/null; then
            fail_reason="oom"
        elif (( cp2k_status == 124 )); then
            fail_reason="timeout"
            echo "FAILED (timeout): {name}" >> "$FAILED_LOG"
        elif (( cp2k_status != 0 )); then
            fail_reason="crash (exit $cp2k_status)"
            echo "FAILED (crash exit=$cp2k_status): {name}" >> "$FAILED_LOG"
        else
            fail_reason="ok"
        fi

        job_elapsed=$(( $(date +%s) - job_start ))
        echo "{name} ${{job_elapsed}}s status=${{fail_reason}}" >> "$TIMES_LOG"
        echo "  finished $(date +%H:%M) (${{job_elapsed}}s, ${{fail_reason}})"

        elapsed=$(( $(date +%s) - start_time ))
        print_eta
        if (( counter % 10 == 0 )); then
            rm -f *.wfn *.wfn.bak-1
            rm -f {cp2k_dir}/*.wfn {cp2k_dir}/*.wfn.bak-1
        fi
        """
        job_blocks.append(block)
    with open(path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(header)
        f.write("\n".join(job_blocks))

def write_all_sp_inputs(selected_frames, cp2k_dir):
    """
    Write CP2K single-point input files and submission scripts.

    REUSE LOGIC (when REUSE_EXISTING_CP2K = True):
    -----------------------------------------------
    Two levels of reuse are checked for each frame:

    Level 1 — direct output check:
      If sp_{sys}_{round}_{i:04d}.out exists and is converged, skip.
      This handles the common case of rerunning after a partial failure.

    Level 2 — geometry hash deduplication:
      A geometry_index.json file maps MD5(positions + atomic numbers) to
      job names.  If a previous job (same or different round) computed the
      exact same geometry, its output is symlinked under the new job name
      so the parser finds it without running CP2K again.
      This catches NEB images that appear in multiple rounds or dissolved
      frames that happen to be identical.

    submit_all.sh    — every job (use for forced full rerun)
    submit_missing.sh — only genuinely new jobs (use day-to-day)
    """
    os.makedirs(cp2k_dir, exist_ok=True)

    # Load existing hash index (may be empty on first run)
    geom_index   = _load_geometry_index(cp2k_dir) if REUSE_EXISTING_CP2K else {}

    pool_hashes = set()
    if REUSE_EXISTING_CP2K and Path(POOL_FILE).exists():
        try:
            pool_frames = read(POOL_FILE, index=":")
            pool_hashes = {get_atoms_hash(a) for a in pool_frames}
            print(f"[→] Loaded {len(pool_hashes)} geometry hashes from master pool")
        except Exception as e:
            print(f"Could not load master pool hashes: {e}")
                
    all_jobs     = []
    missing_jobs = []
    reused_direct  = 0
    reused_hash    = 0
    reused_pool    = 0   
    new_index      = dict(geom_index)   # will be updated and saved at the end

    for i, atoms in enumerate(selected_frames):
        sys_type = atoms.info.get("system_type", "unknown")
        name     = f"sp_{sys_type}_r{ROUND}_{i:04d}"
        inp      = write_cp2k_sp(atoms, name, cp2k_dir)
        all_jobs.append((name, inp))
        expected_out = Path(cp2k_dir) / f"{name}.out"

        if REUSE_EXISTING_CP2K:
            # Level 1: direct .out check
            if _cp2k_output_is_complete(expected_out):
                reused_direct += 1
                new_index[get_atoms_hash(atoms)] = name
                continue

            geom_hash = get_atoms_hash(atoms)
            new_index[geom_hash] = {"name": name, "cp2k_dir": str(cp2k_dir)}

            # Level 2 - checking prior jobs:
            prior_entry = geom_index.get(geom_hash)
            if prior_entry and prior_entry["name"] != name:
                prior_out = Path(prior_entry["cp2k_dir"]) / f"{prior_entry['name']}.out"
                if _cp2k_output_is_complete(prior_out):
                    if not expected_out.exists():
                        expected_out.symlink_to(prior_out.resolve())
                    print(f"  [≡] {name} reuses {prior_entry['name']} from "
                        f"round {prior_entry['cp2k_dir']} (same geometry)")
                    reused_hash += 1
                    continue
            
            # Level 3 - master pool check
            if geom_hash in pool_hashes:
                print(f"  [≡] {name}: geometry already in master pool — skipping")
                reused_pool += 1
                continue
            
        missing_jobs.append((name, inp))
        new_index[get_atoms_hash(atoms)] = name

    # Persist updated index
    if REUSE_EXISTING_CP2K:
        _save_geometry_index(cp2k_dir, new_index)

    n_total   = len(all_jobs)
    n_skipped = reused_direct + reused_hash

    _write_submission_script(
        Path(cp2k_dir) / "submit_all.sh",
        all_jobs, cp2k_dir, "ALL jobs", n_total, n_skipped
    )
    _write_submission_script(
        Path(cp2k_dir) / "submit_missing.sh",
        missing_jobs, cp2k_dir, "missing only", n_total, n_skipped
    )

    print(f"\n[✓] CP2K input summary:")
    print(f"    Total frames:          {n_total}")
    if REUSE_EXISTING_CP2K:
        print(f"    Skipped (direct):      {reused_direct}  (.out already present)")
        print(f"    Skipped (hash match):  {reused_hash}  (identical geometry seen before)")
        print(f"    Skipped (in pool):     {reused_pool}  (already in master_train_pool.xyz)")
        print(f"    Need CP2K:             {len(missing_jobs)}")
        print(f"    Inputs written to:     {cp2k_dir}/")
    if missing_jobs:
        print(f"    Run:  bash {cp2k_dir}/submit_missing.sh")
    else:
        print(f"    [✓] Nothing to run — proceed with:")
        print(f"        python active_pipeline.py --parse")

    return all_jobs

# ============================================================
# Step 4 — Parse CP2K outputs → MACE-ready extxyz
# ============================================================

# Conversion constants
HA_TO_EV    = Hartree        # Hartree → eV source https://physics.nist.gov/cgi-bin/cuu/Value?hrev
BOHR_TO_ANG = Bohr       # Bohr → Å source https://conversion.org/length/bohr-atomic-unit-of-length/angstrom
HA_BOHR_TO_EV_ANG = HA_TO_EV / BOHR_TO_ANG   # force unit conversion
HA_BOHR3_TO_EV_ANG3 = HA_TO_EV / (BOHR_TO_ANG ** 3) 

# Note on stress parsing: Need to fix it as it doesnt always detect stress block
def parse_stress_from_out(content):
    """
    Parse the stress tensor from a CP2K .out file.

    CP2K prints the analytical stress tensor in the PRINT / STRESS_TENSOR block:

        STRESS| Analytical stress tensor [GPa]
        STRESS|                        x                   y                   z
        STRESS|      x       -3.510...    -0.046...    -0.003...
        STRESS|      y       -0.046...    -1.367...     0.164...
        STRESS|      z       -0.003...     0.164...    -1.109...

    This is followed by eigenvalue/eigenvector rows that we must not accidentally
    capture.  The STRESS| prefix on every data row makes this unambiguous.

    We first try this GPa block (present in all modern CP2K ENERGY_FORCE runs
    with STRESS_TENSOR ANALYTICAL).  As a fallback we also try the older
    "STRESS TENSOR [a.u.]" plain-text block that appears in some CP2K versions.

    Returns a (3, 3) numpy array in eV/Å³, or None if no block is found.
    The matrix is symmetric; caller should convert to MACE Voigt 6-vector:
        voigt = stress_matrix[[0,1,2,1,0,0], [0,1,2,2,2,1]]
    """
    # -------------------------------------------------------------------
    # Strategy 1: STRESS| Analytical stress tensor [GPa]  (preferred)
    # Matches exactly 3 data rows immediately after the header row.
    # Uses case-insensitive row labels so X/x both work.
    # -------------------------------------------------------------------
    gpa_pat = re.compile(
        r"STRESS\| Analytical stress tensor \[GPa\]\s*\n"
        r"[ \t]*STRESS\|[^\n]*\n"                         # column-header row  (x  y  z)
        r"[ \t]*STRESS\|\s+[xX]\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s*\n"
        r"[ \t]*STRESS\|\s+[yY]\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s*\n"
        r"[ \t]*STRESS\|\s+[zZ]\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)"
    )
    m = gpa_pat.search(content)
    if m:
        vals = [float(m.group(i)) for i in range(1, 10)]
        stress_gpa = np.array(vals).reshape(3, 3)
        # 1 GPa = 1/160.2176634 eV/Å³  (exact SI definition)
        GPa_TO_EV_ANG3 = 1.0 / 160.2176634
        return stress_gpa * GPa_TO_EV_ANG3

    # -------------------------------------------------------------------
    # Strategy 2: STRESS TENSOR [a.u.]  (older CP2K / some versions)
    # -------------------------------------------------------------------
    au_pat = re.compile(
        r"STRESS TENSOR \[a\.u\.\].*?"
        r"X\s+Y\s+Z\s*\n"
        r"\s*X\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s*\n"
        r"\s*Y\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s*\n"
        r"\s*Z\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)\s+([-\d.E+e-]+)",
        re.DOTALL
    )
    m = au_pat.search(content)
    if m:
        vals = [float(m.group(i)) for i in range(1, 10)]
        stress_au = np.array(vals).reshape(3, 3)
        return stress_au * HA_BOHR3_TO_EV_ANG3   # Ha/Bohr³ → eV/Å³

    return None

def audit_and_requeue_specific_jobs(target_keywords, target_round=None, requeue=False):
    """
    Searches for job files across round directories matching provided keywords.
    A) Checks if matching configurations exist in the master pool.
    B) Checks if .inp and .out files exist, tracks duplicates/counts, and diagnoses failures.
    C) Gives an option to append failed/missing jobs to submit_missing.sh.
    """
    from pathlib import Path
    import re
    from ase.io import read

    # Determine directories to scan
    if target_round is not None:
        cp2k_dirs = [Path(f"cp2k_sp_round{target_round}")]
    else:
        # Dynamically find all round directories sorted numerically
        cp2k_dirs = sorted(
            Path(".").glob("cp2k_sp_round*"),
            key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', p.name)]
        )

    if not cp2k_dirs:
        print("[!] No cp2k_sp_round directories found.")
        return

    print("\n" + "="*70)
    print(f"  Targeted Job Audit (Scanning: {[d.name for d in cp2k_dirs]})")
    print("="*70)

    # ----------------------------------------------------------------
    # A) Pre-load Master Pool Configuration Names/System Types
    # ----------------------------------------------------------------
    pool_known_names = set()
    if Path(POOL_FILE).exists():
        try:
            pool = read(POOL_FILE, index=":")
            for atoms in pool:
                if "system_type" in atoms.info:
                    pool_known_names.add(atoms.info["system_type"])
                if "name" in atoms.info:
                    pool_known_names.add(atoms.info["name"])
        except Exception as e:
            print(f"[!] Error reading master pool: {e}")
    else:
        print(f"[!] Master pool {POOL_FILE} not found.")

    # ----------------------------------------------------------------
    # B) Discover and Diagnose matching files across directories
    # ----------------------------------------------------------------
    needs_rerun = []

    for kw in target_keywords:
        print(f"\n--- Searching for keyword: '{kw}' ---")
        matched_jobs_in_kw = []

        # Find all .inp files matching the keyword in any scanned folder
        for cp2k_dir in cp2k_dirs:
            if not cp2k_dir.exists():
                continue
            # Matches any filename containing the keyword
            for inp_file in cp2k_dir.glob(f"*{kw}*.inp"):
                matched_jobs_in_kw.append((inp_file.stem, cp2k_dir, inp_file))

        print(f"--> Found {len(matched_jobs_in_kw)} matching job(s).")

        for job_name, directory, inp_path in matched_jobs_in_kw:
            out_path = directory / f"{job_name}.out"
            
            # Check pool presence (checking both exact and substring match)
            in_pool = job_name in pool_known_names or any(kw in name for name in pool_known_names)
            
            has_out = out_path.exists()
            status = "Unknown"
            failed_midway = False

            if has_out:
                content = out_path.read_text()
                if "SCF run NOT converged" in content:
                    status = "Failed (SCF not converged)"
                    failed_midway = True
                elif not re.search(r"ENERGY\| Total FORCE_EVAL", content):
                    status = "Failed (Crashed or Timeout)"
                    failed_midway = True
                else:
                    status = "Success (Complete)"
            else:
                status = "Missing (.out file not found)"
                failed_midway = True

            print(f"  * Job: {job_name} ({directory.name})")
            print(f"    - In master pool? : {'Yes' if in_pool else 'No'}")
            print(f"    - Has .out file?  : {'Yes' if has_out else 'No'}")
            print(f"    - Status Summary  : {status}")

            if failed_midway:
                needs_rerun.append((job_name, directory, inp_path))

    # ----------------------------------------------------------------
    # C) Append to submission scripts
    # ----------------------------------------------------------------
    if requeue and needs_rerun:
        print(f"\n[→] Requeueing {len(needs_rerun)} failed/missing jobs...")
        
        # Group reruns by directory so we append to each round's respective script
        from collections import defaultdict
        grouped_reruns = defaultdict(list)
        for job_name, directory, inp_path in needs_rerun:
            grouped_reruns[directory].append((job_name, inp_path))

        for directory, jobs_list in grouped_reruns.items():
            submit_path = directory / "submit_missing.sh"
            mode = "a" if submit_path.exists() else "w"
            
            with open(submit_path, mode) as f:
                if mode == "w":
                    f.write("#!/bin/bash\n")
                    f.write(f"# Appended CP2K reruns\n")
                    f.write("timestamp=$(date +%Y%m%d_%H%M)\n\n")
                
                for job_name, inp_path in jobs_list:
                    f.write(f"echo \"Starting targeted rerun: {job_name}\"\n")
                    f.write(f"timeout {CP2K_TIMEOUT} cp2k.ssmp -i {inp_path} -o {directory}/{job_name}.out || echo \"FAILED: {job_name}\" >> {directory}/failed_jobs.txt\n")
            
            print(f"  [✓] Appended {len(jobs_list)} jobs to {submit_path}")
    elif requeue and not needs_rerun:
        print("\n[✓] Audit finished. No failed or missing jobs found to requeue.")

def recover_and_prioritize_missing(target_round=None, n_runs=100):
    """
    Analyzes all previously queued frames that failed or went missing,
    compares them to the current successful POOL_FILE using Farthest 
    Point Sampling (FPS), and generates new CP2K inputs for the most 
    valuable missing structures.

    Args:
        target_round (int, optional): Restrict scan to a specific round.
        n_runs (int): Number of inputs to generate.
    """
    import glob
    from sklearn.preprocessing import normalize
    
    print("\n" + "="*60)
    print(f"  Missing Structure Recovery & FPS Prioritization")
    print(f"  Target Round : {target_round if target_round else 'All Rounds'}")
    print(f"  Requested    : {n_runs} runs")
    print("="*60)
    
    if Path(POOL_FILE).exists():
        pool = read(POOL_FILE, index=":")
        print(f"[→] Loaded {len(pool)} successful frames from master pool.")
    else:
        pool = []
        print(f"[!] Master pool not found. Priority will be based only on diversity of missing frames.")
        
    if target_round is not None:
            al_files = [f"al_selected_round{target_round}.xyz"]
    else:
            al_files = glob.glob("al_selected_round*.xyz")
            
    missing_frames = []
    total_queued = 0
    
    for al_file in al_files:
        if not os.path.exists(al_file):
            print(f"  [!] AL file not found: {al_file} — skipping")
            continue
        
        m = re.search(r"round(\d+)", al_file)
        r_num = int(m.group(1)) if m else ROUND
        frames = read(al_file, index=":")
        total_queued += len(frames)
        
        for i, atoms in enumerate(frames):
            sys_type = atoms.info.get("system_type", "unknown")
            name = f"sp_{sys_type}_r{r_num}_{i:04d}"
            out_file = Path(f"cp2k_sp_round{r_num}") / f"{name}.out"

            # Check if the output is completely missing or failed SCF
            if not _cp2k_output_is_complete(out_file):
                missing_frames.append((name, atoms, r_num))

    if not missing_frames:
        print(f"\n[✓] Excellent! 0 out of {total_queued} expected outputs are missing.")
        return

    print(f"[→] Found {len(missing_frames)} missing/failed outputs out of {total_queued} queued.")

    FEATURE_DIM = 300
    def get_features(atoms_list):
        feats = []
        for a in atoms_list:
            pos = a.get_positions().flatten()
            if len(pos) >= FEATURE_DIM:
                feats.append(pos[:FEATURE_DIM])
            else:
                feats.append(np.pad(pos, (0, FEATURE_DIM - len(pos))))
        return normalize(np.array(feats)) if feats else np.array([])

    print("\n[→] Calculating geometric value (FPS against master pool)...")
    pool_feats = get_features(pool)
    missing_feats = get_features([f[1] for f in missing_frames])

    # Initial ranking: distance to the closest pool frame
    if len(pool_feats) > 0:
        initial_dists = []
        for feat in missing_feats:
            dists = np.linalg.norm(pool_feats - feat, axis=1)
            initial_dists.append(np.min(dists))
        initial_dists = np.array(initial_dists)
    else:
        initial_dists = np.full(len(missing_frames), np.inf)

    # Show top 20 most valuable
    initial_ranking = np.argsort(initial_dists)[::-1]
    print("\n  Top 20 most valuable missing structures (furthest from pool):")
    for i in range(min(20, len(initial_ranking))):
        idx = initial_ranking[i]
        name, _, r_num = missing_frames[idx]
        dist = initial_dists[idx] if len(pool_feats) > 0 else 0.0
        print(f"    {i+1:2d}. {name} (orig round: {r_num}) - Distance: {dist:.4f}")
    
    min_dists = np.copy(initial_dists)
    selected_indices = []

    for _ in range(min(n_runs, len(missing_frames))):
        if len(pool_feats) == 0 and len(selected_indices) == 0:
            best_idx = 0  # Arbitrary start if pool is totally empty
        else:
            best_idx = int(np.argmax(min_dists))

        selected_indices.append(best_idx)

        # Update distances iteratively to ensure diverse sampling
        new_feat = missing_feats[best_idx]
        new_dists = np.linalg.norm(missing_feats - new_feat, axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    prioritized_frames = [missing_frames[idx][1] for idx in selected_indices]
    
    print(f"\n[→] Generating fresh CP2K inputs for the top {len(prioritized_frames)} prioritized frames...")
    print(f"    Target directory: {CP2K_DIR} (Round {ROUND})")
    
    # Clean the system_type so it gets correctly tagged as a retry in the new round
    for atoms in prioritized_frames:
        original_sys = atoms.info.get("system_type", "unknown")
        # Ensure we don't infinitely stack "retry_" tags
        if not original_sys.startswith("retry_"):
            atoms.info["system_type"] = f"retry_{original_sys}"

    # Hand off to your existing writer (it handles submit_missing.sh automatically!)
    write_all_sp_inputs(prioritized_frames, CP2K_DIR)

def parse_cp2k_sp_results(cp2k_dir, selected_frames):
    """
    Parse CP2K .out files and attach DFT energy + forces to the
    corresponding ASE Atoms objects using MACE-Train key names:

        atoms.info["REF_energy"]   — total energy in eV
        atoms.arrays["REF_forces"] — forces in eV/Å, shape (N, 3)

    These are the keys expected by the standard MACE training pipeline.
    """
    patched = []
    failed  = []

    for i, atoms in enumerate(selected_frames):
        sys_type = atoms.info.get("system_type", "unknown")
        name     = f"sp_{sys_type}_r{ROUND}_{i:04d}"
        outfile  = Path(cp2k_dir) / f"{name}.out"
        
        print(f"Parsing {name}.out...")
        if not outfile.exists():
            print(f"  [!] Missing: {outfile}")
            failed.append(i)
            continue

        content = outfile.read_text()

        if "SCF run NOT converged" in content:
            print(f"  [!] SCF not converged: {name} — skipping")
            failed.append(i)
            continue

        # --- Energy ---
        print(f"  Extracting energy from {name}.out...")
        energy_match = re.search(
            r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-\d.]+)",
            content
        )
        if not energy_match:
            print(f"  [!] Energy not found in {name}.out")
            failed.append(i)
            continue

        energy_eV = float(energy_match.group(1)) * HA_TO_EV
        print(f"Energy of {name}: {energy_eV:.6f} eV.")

        # --- Forces ---
        print(f"  Extracting forces from {name}.out...")
        force_block = re.search(
            r"ATOMIC FORCES in \[a\.u\.\](.*?)SUM OF ATOMIC FORCES",
            content, re.DOTALL
        )

        forces_ok = False
        if force_block:
            force_lines = force_block.group(1).strip().split("\n")
            forces = []
            for line in force_lines:
                parts = line.split()
                if len(parts) == 6:   # idx  kind  symbol  fx  fy  fz
                    fx = float(parts[3]) * HA_BOHR_TO_EV_ANG
                    fy = float(parts[4]) * HA_BOHR_TO_EV_ANG
                    fz = float(parts[5]) * HA_BOHR_TO_EV_ANG
                    forces.append([fx, fy, fz])

            if len(forces) == len(atoms):
                atoms.arrays["REF_forces"] = np.array(forces)
                forces_ok = True
            else:
                print(f"  [!] Force count mismatch in {name}: "
                      f"got {len(forces)}, expected {len(atoms)}")

        # --- Attach metadata ---
        atoms.info["REF_energy"] = energy_eV
        atoms.info["al_round"]   = ROUND
        atoms.info["source"]     = "cp2k_sp"

        # --- Stress ---
        stress = parse_stress_from_out(content)
        if stress is not None:
            # Store full 3x3 tensor; MACE training reads REF_stress as Voigt 6-vector.
            # Voigt order: [xx, yy, zz, yz, xz, xy]
            voigt = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
            atoms.info["REF_stress"] = voigt
        else:
            print(f"  [~] No stress tensor found in {name}.out — stress will be absent")

        # Remove live calculator so extxyz writes cleanly
        atoms.calc = None

        if forces_ok:
            print(f" Successfully parsed forces for {name}.")
            print(f" Max force: {np.max(np.linalg.norm(atoms.arrays['REF_forces'], axis=1)):.3f} eV/Å")
            patched.append(atoms)
        else:
            # Keep the frame but flag it — energy is valid, forces are not
            atoms.info["forces_missing"] = True
            patched.append(atoms)
            print(f"  [~] Kept {name} (energy only, forces missing)")

    print(f"\n[✓] Parsed {len(patched)} frames, {len(failed)} failed/missing")
    return patched, failed

def generate_e0s_input(elements,outdir):
    from ase import Atoms
    os.makedirs(outdir, exist_ok=True)
    input_files = []
    
    for sym in elements:
        # Create a single atom in the center of a large box
        atom = Atoms(sym, positions=[(E0_CELL_SIZE/2, E0_CELL_SIZE/2, E0_CELL_SIZE/2)])
        atom.set_cell([E0_CELL_SIZE, E0_CELL_SIZE, E0_CELL_SIZE])
        
        name = f"e0_{sym}"
        inp_path = write_cp2k_sp(atom, name, outdir)
        input_files.append((name, inp_path))
    
    return input_files

def parse_e0_results(outdir, elements):
    """Parses E0 energies and writes them to E0s.json."""
    e0_dict = {}
    print(f"\n[→] Parsing E0 values from {outdir}...")
    
    for sym in elements:
        outfile = Path(outdir) / f"e0_{sym}.out"
        if not outfile.exists():
            print(f"  [!] Missing E0 output for {sym}")
            continue
            
        content = outfile.read_text()
        energy_match = re.search(
            r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-\d.]+)",
            content
        )
        if energy_match:
            # Convert Hartree to eV and store with Atomic Number as key
            e0_dict[Z_MAP[sym]] = float(energy_match.group(1)) * HA_TO_EV
            print(f"  {sym} (Z={Z_MAP[sym]}): {e0_dict[Z_MAP[sym]]:.6f} eV")
            
    if e0_dict:
        with open(E0_JSON, 'w') as f:
            json.dump(e0_dict, f, indent=4)
        print(f"[✓] Saved {E0_JSON}")
    return e0_dict

def build_per_system_training_files(new_frames, out_dir="training_data"):
    """
    Split the newly parsed frames by system_type and write one
    training .extxyz per system.  MACE-Train can load these individually
    or you can concatenate them.

    Also writes a combined file:  {out_dir}/all_systems_round{ROUND}.extxyz
    """
    os.makedirs(out_dir, exist_ok=True)

    by_system = {}
    for atoms in new_frames:
        key = atoms.info.get("system_type", "unknown")
        by_system.setdefault(key, []).append(atoms)

    for sys_name, frames in by_system.items():
        path = os.path.join(out_dir, f"{sys_name}_round{ROUND}.extxyz")
        write(path, frames, format="extxyz")
        print(f"  [✓] {sys_name}: {len(frames)} frames  →  {path}")

    combined = os.path.join(out_dir, f"all_systems_round{ROUND}.extxyz")
    write(combined, new_frames, format="extxyz")
    print(f"  [✓] Combined: {len(new_frames)} frames  →  {combined}")
    return by_system

def is_physically_reasonable(atoms, calc, round_num=1, check_slab_z=False, forces=None):
    """
    Screen a candidate frame for physical reasonableness before
    submitting to CP2K. Returns (is_ok, reason_if_rejected).
    
    Uses three checks:
    1. Minimum interatomic distances (catches overlapping atoms)
    2. Maximum interatomic distances for bonded pairs (catches dissolved atoms)  
    3. MACE per-atom force outlier detection (catches unphysical environments)
    """
    from ase.neighborlist import neighbor_list
    import numpy as np

    # --- Check 1: Minimum distance (atoms too close) ---
    # Get all pairwise distances within 1.0 Å — should be empty
    i, j, d = neighbor_list("ijd", atoms, cutoff=0.7, self_interaction=False)
    if len(d) > 0:
        idx = np.argmin(d)
        return False, f"atoms too close: {atoms.symbols[i[idx]]}-{atoms.symbols[j[idx]]} = {d[idx]:.2f} Å"

    # --- Check 2: Optional low-Z atom check (catches atoms embedded below slab) ---
    # If SLAB_ELEMENT is set in atoms.info, check non-slab atoms aren't buried.
    # This is a generalised version: set atoms.info["slab_element"] = "Pt" (or any element)
    # and atoms.info["slab_z_threshold"] = 5.0 to activate.
    symbols = np.array(atoms.get_chemical_symbols())
    slab_element = atoms.info.get("slab_element", None)
    z_threshold  = float(atoms.info.get("slab_z_threshold", 5.0))
    if slab_element:
        symbols_arr  = np.array(atoms.get_chemical_symbols())
        positions_   = atoms.get_positions()
        non_slab     = (symbols_arr != slab_element)
        if np.any(non_slab):
            min_z = np.min(positions_[non_slab, 2])
            if min_z < z_threshold:
                return False, f"Non-{slab_element} atom below Z-threshold {z_threshold:.1f} Å: min_z={min_z:.2f} Å"

    # --- Check 3: MACE per-atom force outlier ---
    # Scale the ceiling with round number — be stricter early on
    force_ceilings = {1: 8.0, 2: 10.0, 3: 12.0, 4: 20.0}
    ceiling = force_ceilings.get(round_num, 15.0)

    # Use a copy to avoid attaching the calc to the original object permanently
    if forces is None:
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        forces = atoms_copy.get_forces()

    force_mags = np.linalg.norm(forces, axis=1)
    
    max_f = np.max(force_mags)
    mean_f = np.mean(force_mags)
    
    if max_f > ceiling:
        worst_atom = symbols[np.argmax(force_mags)]
        return False, f"MACE force {max_f:.2f} eV/Å on {worst_atom} exceeds ceiling {ceiling}"
    
    if max_f > 5 * mean_f and max_f > 3.0:
        worst_atom = symbols[np.argmax(force_mags)]
        return False, f"Force outlier: {worst_atom} ({max_f:.2f} eV/Å) vs mean ({mean_f:.2f} eV/Å)"

    return True, "ok"

# ============================================================
# Pre-CP2K triage: capped relaxation for pathological initial forces
# ============================================================
def relax_pathological_frames(frames, calc, trigger_force=GEOOPT_TRIGGER_FORCE,
                               max_steps=GEOOPT_MAX_STEPS, fmax_target=GEOOPT_FMAX_TARGET):
    """
    For any frame whose initial MACE force magnitude exceeds `trigger_force`,
    run a short, CAPPED relaxation to remove pathological overlaps/clashes
    before paying for a CP2K single-point.

    This is deliberately NOT a full optimisation:
      - max_steps caps how far the geometry can move
      - fmax_target is loose (eV/Å, not meV/Å) -- we're aiming for
        "no longer exploding", not "sitting in a minimum"
    Fully converging would relax the frame toward whatever the *current*
    model already considers low-energy, which destroys the off-equilibrium,
    informative character that made it worth selecting in the first place.
    Only frames that actually improve are kept relaxed; if the optimiser
    doesn't help within the step budget, the original geometry is kept
    as-is and flagged so you can inspect it manually.

    Tags written to atoms.info: pre_relaxed, pre_relax_steps,
    pre_relax_max_force_before, pre_relax_max_force_after.
    """
    from ase.optimize import FIRE
    import numpy as np

    n_triggered = 0
    n_improved  = 0

    for atoms in frames:
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        try:
            f0 = atoms_copy.get_forces()
        except Exception as e:
            print(f"    [!] Triage: could not evaluate initial forces, skipping check: {e}")
            continue

        max_f0 = float(np.max(np.linalg.norm(f0, axis=1)))
        if max_f0 <= trigger_force:
            continue

        n_triggered += 1
        opt = FIRE(atoms_copy, logfile=None)
        try:
            opt.run(fmax=fmax_target, steps=max_steps)
            f_final = atoms_copy.get_forces()
            max_f_final = float(np.max(np.linalg.norm(f_final, axis=1)))
        except Exception as e:
            print(f"    [!] Triage relaxation failed for one frame, keeping original "
                  f"geometry (initial max_f={max_f0:.2f} eV/Å): {e}")
            atoms.info["pre_relax_failed"] = True
            continue

        if max_f_final < max_f0:
            atoms.set_positions(atoms_copy.get_positions())
            atoms.info["pre_relaxed"] = True
            atoms.info["pre_relax_steps"] = int(opt.nsteps)
            atoms.info["pre_relax_max_force_before"] = max_f0
            atoms.info["pre_relax_max_force_after"]  = max_f_final
            n_improved += 1
            print(f"    [⚙] Pre-relaxed: {max_f0:.2f} → {max_f_final:.2f} eV/Å "
                  f"in {opt.nsteps} steps  ({atoms.info.get('system_type','?')})")
        else:
            atoms.info["pre_relax_no_improvement"] = True
            print(f"    [!] Triage didn't help within {max_steps} steps "
                  f"({max_f0:.2f} → {max_f_final:.2f} eV/Å) — keeping original geometry, "
                  f"consider rejecting this frame manually "
                  f"({atoms.info.get('system_type','?')})")

    print(f"  [✓] Pre-CP2K triage: {n_triggered} frame(s) exceeded "
          f"{trigger_force} eV/Å, {n_improved} improved by relaxation")
    return frames

# ============================================================
# REICO: random "imaginary chemical" box generation
# ============================================================
def create_random_box(elements, n_atoms, vol_per_atom=REICO_VOL_PER_ATOM,
                       min_dist_scale=REICO_MIN_DIST_SCALE, max_attempts=300):
    """
    Build one small periodic box containing `n_atoms` atoms drawn with
    repetition from `elements`. Two things matter here that a naive random
    placement gets wrong:

    1. Box volume scales with n_atoms (via vol_per_atom) so you get a dense,
       chemically relevant local environment -- a fixed large cell mostly
       just produces isolated atoms drifting in vacuum, which doesn't teach
       the model anything about short-range repulsion.
    2. The minimum allowed distance is per ELEMENT PAIR (scaled sum of
       covalent radii), not one global number. A single global cutoff is
       wrong in both directions: too loose for H-H, way too tight for
       Pt-Pt (real Pt-Pt is >2.6 Å; 1.7 Å would itself be a pathological
       clash, exactly what this whole exercise is trying to avoid creating).
    """
    from ase import Atoms, Atom
    from ase.data import covalent_radii, atomic_numbers as ase_Z
    from ase.geometry import get_distances

    edge = (n_atoms * vol_per_atom) ** (1 / 3)
    cell = [edge, edge, edge]
    chosen = list(np.random.choice(elements, size=n_atoms))

    atoms = Atoms(cell=cell, pbc=True)
    for el in chosen:
        placed = False
        for _ in range(max_attempts):
            pos = np.random.uniform(0, edge, size=3)
            if len(atoms) == 0:
                atoms.append(Atom(el, pos))
                placed = True
                break

            existing_syms = atoms.get_chemical_symbols()
            min_allowed = np.array([
                min_dist_scale * (covalent_radii[ase_Z[el]] + covalent_radii[ase_Z[s]])
                for s in existing_syms
            ])
            _, d = get_distances([pos], atoms.get_positions(), cell=atoms.get_cell(), pbc=True)
            if np.all(d[0] >= min_allowed):
                atoms.append(Atom(el, pos))
                placed = True
                break

        if not placed:
            raise ValueError(
                f"Could not place {el} in a {edge:.1f} Å box after {max_attempts} "
                f"attempts -- box is too small/dense for {n_atoms} atoms, try "
                f"raising REICO_VOL_PER_ATOM"
            )
    return atoms

# ============================================================
# Main entry points
# ============================================================
from mace.calculators import MACECalculator 
def run_round():
    """Load MACE candidate files, select uncertain frames, write CP2K inputs."""
    print(f"\n{'='*60}")
    print(f"  Active Learning Round {ROUND}")
    print(f"  Model: {MODEL_PATH}")
    print(f"{'='*60}\n")
    skipped_unphysical = 0
    
    # ---- NEB / GeoOpt candidates ----
    all_candidates = load_candidates(AL_INPUT_DIR)
    
    print(f"\n[→] Re-scoring {len(all_candidates)} NEB/GeoOpt frames with MACE...")
    all_candidates = rescore_with_mace(all_candidates, MODEL_PATH)
 
    # Load global geometry index once — used to check what's already done
    geom_index = _load_geometry_index(CP2K_DIR) if REUSE_EXISTING_CP2K else {}
    def _already_computed(atoms):
        """Return True if this geometry already has a complete CP2K output."""
        # Level 1: direct .out check using expected job name
        # We don't know the name yet at selection time, so use hash only
        geom_hash = get_atoms_hash(atoms)
        entry = geom_index.get(geom_hash)
        if entry:
            out = Path(entry["cp2k_dir"]) / f"{entry['name']}.out"
            if _cp2k_output_is_complete(out):
                return True
        return False

    # Score all candidates by MACE force magnitude
    print("\nScoring all candidates by MACE max force...")
    scored = []
    for atoms in all_candidates:
        f = None
        if atoms.calc is not None:
            try:
                f = atoms.get_forces()
                score = float(np.max(np.linalg.norm(f, axis=1)))
            except Exception:
                score = 0.0
        else:
            score = 0.0
        scored.append((score, atoms, f))

    # Sort by descending force magnitude — most uncertain first
    scored.sort(key=lambda x: x[0], reverse=True)

    # Walk down the ranked list, skipping already-computed geometries,
    # until we have N_SELECT_TOTAL frames that genuinely need CP2K
    selected = []
    skipped_computed = 0
    skipped_seen = set()  # hashes already in selected (avoid duplicates within batch)

    print(f"\n[→] Selecting up to {N_SELECT_TOTAL} frames that need CP2K...")
    
    calc_mace = MACECalculator(model_paths=MODEL_PATH, device="cuda", default_dtype="float32")
    if APPLY_D3:
        print(f"  [→] D3 dispersion correction will be applied to MACE scores.")
        calc_DFT = TorchDFTD3Calculator(
                    device="cuda",
                    damping="bj",
                    xc=cfg.get("dispersion_xc", "pbe"),
                    cutoff=cfg.get("dispersion_cutoff", 40.0),
                )
        calc = SumCalculator([calc_mace, calc_DFT])    
    else:
        print(f"  [→] No D3 dispersion correction applied to MACE scores.")
        calc = calc_mace
    for score, atoms, forces in scored:
        if len(selected) >= N_SELECT_TOTAL:
            break

        geom_hash = get_atoms_hash(atoms)

        # Skip if identical geometry already in this batch
        if geom_hash in skipped_seen:
            continue

        # Skip if already computed in any previous round
        if _already_computed(atoms):
            skipped_computed += 1
            continue

        # Skip if below force threshold (only if we have enough candidates above it)
        is_ok, reason = is_physically_reasonable(atoms, calc, round_num=ROUND, forces=forces)
        if not is_ok:
            print(f"  [✗] Skipped (unphysical): "
                f"{atoms.info.get('system_type','?')}  — {reason}")
            skipped_unphysical += 1
            continue
        
        skipped_seen.add(geom_hash)
        selected.append(atoms)
        print(f"  [+] Selected frame {len(selected):3d}/{N_SELECT_TOTAL}  "
              f"score={score:.3f} eV/Å  "
              f"system={atoms.info.get('system_type','?')}")

    print(f"\n  [✓] Selected {len(selected)} frames for CP2K "
          f"    {skipped_computed} already-computed geometries skipped"
          f"    Unphysical/noisy : {skipped_unphysical}"
          f"    Remaining pool   : {len(scored) - len(selected) - skipped_computed - skipped_unphysical}")

    if len(selected) < N_SELECT_TOTAL:
        print(f"  [!] Warning: only found {len(selected)} new frames — "
              f"  candidate pool may be exhausted. Consider running more GeoOpts/NEBs.")
    if EXTERNAL_DATASETS:
        print(f"\n[→] Sampling external dataset structures for CP2K recalculation...")

        for src_name, src_cfg in EXTERNAL_SOURCES.items():
            src_path = src_cfg["path"]
            n_samples = src_cfg["n_samples"]

            if not os.path.exists(src_path):
                print(f"  [!] File not found, skipping: {src_path}")
                continue

            traj = read(src_path, index=":")
            print(f"  [{src_name}] Loaded {len(traj)} frames from {Path(src_path).name}")

            # Strip any existing calculator/energy info — CP2K will provide new labels
            # This is critical: you don't want OC25/MPtrj energies leaking into
            # the CP2K input or confusing the parser later
            clean_traj = []
            for atoms in traj:
                atoms_clean = atoms.copy()
                atoms_clean.calc = None
                # Remove energy/forces keys from info so they don't 
                # conflict with CP2K output keys
                for key in ("energy", "forces", "stress", "corrected_total_energy",
                            "REF_energy", "REF_forces", "REF_stress"):
                    atoms_clean.info.pop(key, None)
                    if key in atoms_clean.arrays:
                        del atoms_clean.arrays[key]
                clean_traj.append(atoms_clean)

            # FPS diversity sampling — reuses your existing function
            sampled = fps_sample_md_trajectory(clean_traj, n_samples, src_name)

            # Screen for overlapping atoms / MACE force outliers, but skip
            # Check 1.5 (the Pt-slab z-threshold) -- it assumes a slab
            # geometry that doesn't apply to OC25/MPtrj structures.
            screened = []
            n_rejected = 0
            for atoms in sampled:
                is_ok, reason = is_physically_reasonable(
                    atoms, calc, round_num=ROUND, check_slab_z=False
                )
                if is_ok:
                    screened.append(atoms)
                else:
                    n_rejected += 1
                    print(f"    [✗] {src_name}: rejected — {reason}")

            selected.extend(screened)
            print(f"  [✓] {src_name}: added {len(screened)} frames "
                f"({n_rejected} rejected by physicality screen) "
                f"(total selected so far: {len(selected)})")

    if REICO_SAMPLEING == True:
        print(f"\n[→] REICO: generating {REICO_NUM} random imaginary-chemical boxes...")
        unique_elements = sorted(set(
            sym for atoms in selected for sym in atoms.get_chemical_symbols()
        ))
        print(f"Creating REICO samples with {unique_elements}")
        if not unique_elements:
            print("  [!] REICO: no elements found in selected pool yet, skipping.")
        else:
            reico_frames = []
            n_failed = 0
            for _ in range(REICO_NUM):
                n_atoms = int(np.random.randint(REICO_MIN_ATOMS, REICO_MAX_ATOMS + 1))
                try:
                    box = create_random_box(unique_elements, n_atoms)
                except ValueError as e:
                    n_failed += 1
                    print(f"    [!] REICO: {e}")
                    continue
                box.info["system_type"] = "reico_random"
                reico_frames.append(box)

            print(f"  [✓] REICO: generated {len(reico_frames)}/{REICO_NUM} boxes "
                  f"({n_failed} failed placement)")

            # These start from raw random placement, so they're guaranteed
            # to need relaxing -- trigger_force=0.0 forces it every time,
            # with a bigger step budget than the general triage gets since
            # they start further from anything reasonable.
            reico_frames = relax_pathological_frames(
                reico_frames, calc, trigger_force=0.0,
                max_steps=60, fmax_target=GEOOPT_FMAX_TARGET
            )

            n_rejected = 0
            for box in reico_frames:
                is_ok, reason = is_physically_reasonable(
                    box, calc, round_num=ROUND, check_slab_z=False
                )
                if is_ok:
                    selected.append(box)
                else:
                    n_rejected += 1
                    print(f"    [✗] reico_random: rejected — {reason}")

            print(f"  [✓] REICO: added {len(reico_frames) - n_rejected} frames "
                  f"({n_rejected} rejected by physicality screen) "
                  f"(total selected so far: {len(selected)})")
    
    print(f"\n[✓] Total frames queued for CP2K: {len(selected)}")
    
    for atoms in selected:
        atoms.calc = None

    if GEOOPT_TRIGGER == True:
        print(f"\n[→] Pre-CP2K triage: checking for pathological initial forces "
            f"(trigger > {GEOOPT_TRIGGER_FORCE} eV/Å)...")
        selected = relax_pathological_frames(selected, calc)

    e0_inputs = []  # ← fix for the UnboundLocalError
    if not os.path.exists(E0_JSON):
        elements_needed = set()
        for atoms in selected:
            elements_needed.update(atoms.get_chemical_symbols())
        print(f"\n[→] {E0_JSON} not found. Generating E0 inputs for: {elements_needed}")
        e0_inputs = generate_e0s_input(sorted(list(elements_needed)), E0_DIR)

    write_all_sp_inputs(selected, CP2K_DIR)

    selected_path = f"al_selected_round{ROUND}.xyz"
    write(selected_path, selected, format="extxyz")
    print(f"[✓] Selected frames saved to: {selected_path}")

    if e0_inputs:
        submit_path = Path(CP2K_DIR) / "submit_all.sh"
        with open(submit_path, "a") as f:
            f.write("\n# --- E0 Isolated Atom Calculations ---\n")
            for name, inp in e0_inputs:
                out = Path(E0_DIR) / f"{name}.out"
                f.write(f"cp2k.ssmp -i {inp} -o {out}\n")
                
    print(f"\nNext steps:")
    print(f"  1. Run CP2K:  bash {CP2K_DIR}/submit_all.sh")
    print(f"  2. Parse:     python active_pipeline.py --parse")
    print(f"  3. Re-try failed jobs, then:  python active_pipeline.py --reparse")
    print(f"  4. Retrain:   update ROUND to {ROUND+1} in both scripts, "
          f"then run train_multiple.sh")
 
 
def parse_and_update():
    """Call after CP2K jobs finish: parse outputs, write MACE training data."""
    selected_path = f"al_selected_round{ROUND}.xyz"
    if not os.path.exists(selected_path):
        raise FileNotFoundError(
            f"Cannot find {selected_path}. "
            "Run  python active_pipeline.py  first (without --parse)."
        )
 
    selected = read(selected_path, index=":")
    
    if not os.path.exists(E0_JSON):
        print(f"\n[→] {E0_JSON} not found. Parsing E0 results...")
        elements_needed = set()
        for atoms in selected:
            elements_needed.update(atoms.get_chemical_symbols())
        parse_e0_results(E0_DIR, sorted(list(elements_needed)))
        print(f"E0s for each element are now saved in {E0_JSON}")
    else:
        print(f"\n[✓] Found existing {E0_JSON}, skipping E0 parsing.")
    
    new_frames, failed = parse_cp2k_sp_results(CP2K_DIR, selected)
 
    _write_outputs(new_frames, failed)
    return new_frames

def reparse_failed():
    """
    Re-parse only the frames that previously failed.
 
    Reads failed frame indices from the selected file by matching job names
    found in failed_jobs.txt (written by your  timeout ... || echo FAILED >>
    wrapper).  Merges any newly-recovered frames into the master pool and the
    round archive without re-processing frames that already succeeded.
 
    Usage:  python active_pipeline.py --reparse
    """
    selected_path = f"al_selected_round{ROUND}.xyz"
    if not os.path.exists(selected_path):
        raise FileNotFoundError(
            f"Cannot find {selected_path}. "
            "Cannot determine which frames to re-parse."
        )
 
    # ---- Work out which frame indices to retry ----
    # Strategy A: use failed_jobs.txt written by the timeout wrapper
    failed_indices = []
    if os.path.exists(FAILED_LOG):
        with open(FAILED_LOG) as f:
            for line in f:
                # Line format: "FAILED: sp_Dry0.0Pt_r1_0003"
                m = re.search(r"sp_\S+_r\d+_(\d+)", line)
                if m:
                    failed_indices.append(int(m.group(1)))
        print(f"[→] Found {len(failed_indices)} failed job indices in {FAILED_LOG}")
    else:
        # Strategy B: check which .out files are still missing or unconverged
        print(f"[!] {FAILED_LOG} not found — scanning CP2K output directory instead")
        all_selected = read(selected_path, index=":")
        for i, atoms in enumerate(all_selected):
            sys_type = atoms.info.get("system_type", "unknown")
            name     = f"sp_{sys_type}_r{ROUND}_{i:04d}"
            outfile  = Path(CP2K_DIR) / f"{name}.out"
            if not outfile.exists():
                failed_indices.append(i)
            elif "SCF run NOT converged" in outfile.read_text():
                failed_indices.append(i)
        print(f"[→] Found {len(failed_indices)} missing/unconverged outputs")
 
    if not failed_indices:
        print("[✓] No failed frames found — nothing to reparse.")
        return
 
    all_selected    = read(selected_path, index=":")
    failed_frames   = [all_selected[i] for i in failed_indices
                       if i < len(all_selected)]
 
    print(f"[→] Attempting to parse {len(failed_frames)} previously-failed frames...")
 
    # Temporarily renumber so parse_cp2k_sp_results finds the right filenames
    # We need the original index to reconstruct job names correctly.
    # Re-implement a targeted parse here rather than calling the bulk function.
    recovered = []
    still_failed = []
 
    HA_TO_EV    = Hartree       
    BOHR_TO_ANG = Bohr         
    HA_BOHR_TO_EV_ANG = HA_TO_EV / BOHR_TO_ANG   # force unit conversion
 
    for orig_idx in failed_indices:
        if orig_idx >= len(all_selected):
            continue
        atoms    = all_selected[orig_idx].copy()
        sys_type = atoms.info.get("system_type", "unknown")
        name     = f"sp_{sys_type}_r{ROUND}_{orig_idx:04d}"
        outfile  = Path(CP2K_DIR) / f"{name}.out"
 
        if not outfile.exists():
            print(f"  [!] Still missing: {name}.out")
            still_failed.append(orig_idx)
            continue
 
        content = outfile.read_text()
 
        if "SCF run NOT converged" in content:
            print(f"  [!] Still unconverged: {name}")
            still_failed.append(orig_idx)
            continue
 
        energy_match = re.search(
            r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-\d.]+)",
            content
        )
        if not energy_match:
            print(f"  [!] No energy in {name}.out")
            still_failed.append(orig_idx)
            continue
 
        energy_eV = float(energy_match.group(1)) * HA_TO_EV
 
        force_block = re.search(
            r"ATOMIC FORCES in \[a\.u\.\](.*?)SUM OF ATOMIC FORCES",
            content, re.DOTALL
        )
        forces_ok = False
        if force_block:
            forces = []
            for line in force_block.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) == 6:
                    forces.append([
                        float(parts[3]) * HA_BOHR_TO_EV_ANG,
                        float(parts[4]) * HA_BOHR_TO_EV_ANG,
                        float(parts[5]) * HA_BOHR_TO_EV_ANG,
                    ])
            if len(forces) == len(atoms):
                atoms.arrays["REF_forces"] = np.array(forces)
                forces_ok = True
 
        atoms.info["REF_energy"] = energy_eV
        atoms.info["al_round"]   = ROUND
        atoms.info["source"]     = "cp2k_sp"
        atoms.calc = None
        
        # --- Stress ---
        stress = parse_stress_from_out(content)
        if stress is not None:
            voigt = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
            atoms.info["REF_stress"] = voigt
            
        if not forces_ok:
            atoms.info["forces_missing"] = True
            print(f"  [~] Recovered {name} (energy only)")
 
        recovered.append(atoms)
        print(f"  [✓] Recovered: {name}  E = {energy_eV:.4f} eV")
 
    print(f"\n[✓] Recovered {len(recovered)} frames, "
          f"{len(still_failed)} still failed")
 
    if not recovered:
        print("[!] Nothing new to add — exiting.")
        return
 
    # Merge recovered frames into the existing round archive and master pool
    archive_path = f"al_cp2k_results_round{ROUND}.extxyz"
    if os.path.exists(archive_path):
        existing = read(archive_path, index=":")
        existing.extend(recovered)
        write(archive_path, existing, format="extxyz")
        print(f"[✓] Round archive updated: {len(existing)} total frames  →  {archive_path}")
    else:
        write(archive_path, recovered, format="extxyz")
 
    if os.path.exists(POOL_FILE):
        pool = read(POOL_FILE, index=":")
    else:
        pool = []
    pool.extend(recovered)
    write(POOL_FILE, pool, format="extxyz")
    print(f"[✓] Master pool updated: {len(pool)} total frames  →  {POOL_FILE}")
 
    # Rewrite the failed log with only the frames that are still failing
    if still_failed and os.path.exists(FAILED_LOG):
        with open(FAILED_LOG, "w") as f:
            for idx in still_failed:
                sys_type = all_selected[idx].info.get("system_type", "unknown")
                f.write(f"FAILED: sp_{sys_type}_r{ROUND}_{idx:04d}\n")
        print(f"[✓] Updated {FAILED_LOG} — {len(still_failed)} jobs still pending")

def parse_cell_from_out(content):
    """
    Parse the 3x3 cell matrix from a CP2K .out file.
    Reads the 'CELL|' block (not CELL_TOP or CELL_REF).
    Returns a 3x3 numpy array in Angstrom, or None on failure.
    """
    cell = np.zeros((3, 3))
    labels = {"a": 0, "b": 1, "c": 2}

    for vec_label, row_idx in labels.items():
        # Matches: CELL| Vector a [angstrom]:      11.099     0.000     0.000
        pattern = (
            rf"^ CELL\| Vector {vec_label} \[angstrom\]:\s+"
            r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        )
        m = re.search(pattern, content, re.MULTILINE)
        if not m:
            return None
        cell[row_idx] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]

    return cell

def parse_positions_from_out(content):
    """
    Parse atomic symbols and positions from the CP2K .out file.

    Looks for the block:
        MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM
        ...
        Atom Kind Element         X             Y             Z       Z(eff)     Mass
           1    1 Pt   78     11.067479 ...

    Returns (symbols: list[str], positions: np.ndarray shape (N,3)), or (None, None).
    """
    # Find the coordinates block — take the LAST occurrence (final geometry)
    block_pattern = re.compile(
        r"MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM.*?"
        r"Atom\s+Kind\s+Element.*?\n"
        r"(.*?)\n\s*\n",   # rows until blank line
        re.DOTALL
    )
    matches = list(block_pattern.finditer(content))
    if not matches:
        return None, None

    # Use the last match — for GEO_OPT this is the final structure;
    # for ENERGY_FORCE there is only one.
    block = matches[-1].group(1)

    symbols   = []
    positions = []
    for line in block.strip().split("\n"):
        parts = line.split()
        # Expected columns: idx  kind  symbol  atomic_number  x  y  z  z_eff  mass
        if len(parts) >= 7:
            try:
                # Column 2 (0-indexed) is the element symbol
                sym = parts[2]
                x   = float(parts[4])
                y   = float(parts[5])
                z   = float(parts[6])
                symbols.append(sym)
                positions.append([x, y, z])
            except (ValueError, IndexError):
                continue

    if not symbols:
        return None, None

    return symbols, np.array(positions)
 
def parse_all_cp2k_outputs(target_round=None):
    """
    Scan ALL cp2k_sp_round* directories. For each complete .out file,
    read atomic positions from the matching .out file, parse energy and
    forces from the .out file, and add unique frames to master_train_pool.xyz.

    No dependency on al_selected_round{N}.xyz — positions come from the
    .out file that CP2K actually ran, so this is fully self-contained.

    Usage:
        python active_pipeline.py round 1 --parse-all
    """
    import glob

    print("\n" + "="*60)
    print(f"  CP2K Output Scan {'(Round ' + str(target_round) + ' Only)' if target_round else 'All Rounds'}")
    print("="*60)

    try: 
        with open(E0_JSON, "r") as file:
            E0s_ref = {int(k): v for k, v in json.load(file).items()}
            print(f"Your E0s are {E0s_ref}")
    except FileNotFoundError:
        print(f"Error: The file '{E0_JSON}' could not be found.")
        E0s_ref = {}  # Provide a fallback empty dictionary so the rest of the script doesn't break
    
    # ----------------------------------------------------------------
    # Step 1 — Load existing master pool and build hash index
    # ----------------------------------------------------------------
    if Path(POOL_FILE).exists():
        print(f"\n[→] Loading existing master pool: {POOL_FILE}")
        pool = read(POOL_FILE, index=":")
        print(f"    {len(pool)} frames currently in pool")
    else:
        print(f"\n[→] No existing master pool — will create {POOL_FILE}")
        pool = []

    pool_hashes = {get_atoms_hash(a) for a in pool}
    print(f"    {len(pool_hashes)} unique geometries already in pool")

    # ----------------------------------------------------------------
    # Step 2 — Find all cp2k_sp_round* directories
    # ----------------------------------------------------------------
    if target_round is not None:
        round_dirs = sorted(glob.glob(f"cp2k_sp_round{target_round}"))
    else:
        round_dirs = sorted(glob.glob("cp2k_sp_round*"))

    if not round_dirs:
        print("\n[!] No cp2k_sp_round* directories found.")
        return

    print(f"\n[→] Found {len(round_dirs)} round directories:")
    for d in round_dirs:
        n_out = len(list(Path(d).glob("sp_*.out")))
        n_inp = len(list(Path(d).glob("sp_*.inp")))
        print(f"    {d}/  ({n_inp} .inp  |  {n_out} .out)")

    # ----------------------------------------------------------------
    # Step 3 — Parse each .inp + .out pair
    # ----------------------------------------------------------------
    new_frames   = []
    already_have = 0
    parse_failed = 0

    for cp2k_dir in round_dirs:
        m = re.search(r"cp2k_sp_round(\d+)", cp2k_dir)
        round_num = int(m.group(1)) if m else 0

        out_files = sorted(Path(cp2k_dir).glob("sp_*.out"))
        print(f"\n[→] Round {round_num}: scanning {len(out_files)} .out files...")

        n_symlinks   = 0
        n_incomplete = 0
        n_excluded   = 0
        n_too_large  = 0
        n_dup        = 0
        for out_path in out_files:

            # Skip symlinks — already counted under their source name
            if out_path.is_symlink():
                n_symlinks += 1
                continue

            # Check .out is complete before doing any work
            if not _cp2k_output_is_complete(out_path):
                n_incomplete += 1
                continue
            
            # --------------------------------------------------------
            # Parse positions + species from .inp
            # --------------------------------------------------------
            out_content = out_path.read_text()

            # Extract cell: ABC a b c
            cell_matrix = parse_cell_from_out(out_content)
            if cell_matrix is None:
                # Diagnose: show what CELL| lines are actually present
                cell_lines = [l.strip() for l in out_content.splitlines()
                              if "CELL|" in l and "Vector" in l]
                print(f"  [!] Could not parse cell from {out_path.name} — skipping")
                if cell_lines:
                    print(f"      (Found CELL| lines: {cell_lines[:3]})")
                else:
                    print(f"      (No 'CELL| Vector' lines found — check CP2K output format)")
                parse_failed += 1
                continue

            # Parse positions from .out
            symbols, positions = parse_positions_from_out(out_content)
            if symbols is None:
                has_qs = "MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM" in out_content
                print(f"  [!] Could not parse positions from {out_path.name} — skipping")
                print(f"      (QUICKSTEP coord block present: {has_qs})")
                parse_failed += 1
                continue

            # Build ASE Atoms from .out data
            from ase import Atoms as AseAtoms
            atoms = AseAtoms(
                symbols=symbols,
                positions=positions,
                cell=cell_matrix,
                pbc=True
            )

            # system_type from filename as before
            stem = out_path.stem
            m2 = re.match(r"sp_(.+?)_r\d+(?:_\d+)?$", stem)
            if not m2:
                # Filename doesn't match expected sp_{sys}_r{N}_{i:04d} pattern
                print(f"  [!] Filename {stem} doesn't match sp_*_rN_NNNN pattern — skipping")
                parse_failed += 1
                continue
            sys_type = m2.group(1)
            if _is_excluded(sys_type):
                n_excluded += 1
                continue
            atoms.info["system_type"] = sys_type

            # Check hash against pool before parsing .out
            geom_hash = get_atoms_hash(atoms)
            if geom_hash in pool_hashes:
                n_dup += 1
                already_have += 1
                continue
                        
            #Check the cell isnt too big for memory
            # Adjust MAX_ATOMS at the top of this file.
            n_atoms = len(atoms)
            if n_atoms > MAX_ATOMS:
                print(f"  [!] {out_path.name}: {n_atoms} atoms exceeds limit "
                      f"({MAX_ATOMS}) — skipping")
                n_too_large += 1
                parse_failed += 1
                continue
            # --------------------------------------------------------
            # Parse energy + forces from .out
            # --------------------------------------------------------
            energy_match = re.search(
                r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-\d.]+)",
                out_content
            )
            if not energy_match:
                print(f"  [!] No energy in {out_path.name} — skipping")
                parse_failed += 1
                continue

            energy_eV    = float(energy_match.group(1)) * HA_TO_EV
            symbols_list = atoms.get_chemical_symbols()
            symbols_set  = set(symbols_list)
            pt_count     = symbols_list.count("Pt")
            
            # Validate residual — catch bad SCF before accepting
            e_ref    = sum(E0s_ref.get(z, 0.0) for z in atoms.numbers)
            residual = (energy_eV - e_ref) / len(atoms)

            if pt_count > 3:                                         # Pt slab
                res_lo, res_hi = -20.0, 10.0
            elif pt_count > 0:                                       # dissolved Pt
                res_lo, res_hi = -20.0, 10.0
            elif "P" in symbols_set or "N" in symbols_set:
                res_lo, res_hi = -15.0, 7.0
            elif any(s in symbols_set for s in ("F", "S", "C")):    # Nafion
                res_lo, res_hi = -10.0, 10.0
            elif symbols_set <= {"H", "O"}:                         # bulk water
                res_lo, res_hi = -20.0, 10.0
            else:                                                    # fallback
                res_lo, res_hi = -10.0, 5.0               

            if not (res_lo < residual < res_hi):
                print(f"  [!] {out_path.name}: residual={residual:.2f} eV/atom "
                      f"(allowed {res_lo} to {res_hi}) — skipping")
                parse_failed += 1
                continue

            # Parse forces
            force_block = re.search(
                r"ATOMIC FORCES in \[a\.u\.\](.*?)SUM OF ATOMIC FORCES",
                out_content, re.DOTALL
            )
            forces_ok = False
            if force_block:
                forces = []
                for line in force_block.group(1).strip().split("\n"):
                    parts = line.split()
                    if len(parts) == 6:
                        forces.append([
                            float(parts[3]) * HA_BOHR_TO_EV_ANG,
                            float(parts[4]) * HA_BOHR_TO_EV_ANG,
                            float(parts[5]) * HA_BOHR_TO_EV_ANG,
                        ])
                if len(forces) == len(atoms):
                    atoms.arrays["REF_forces"] = np.array(forces)
                    forces_ok = True
                else:
                    print(f"  [!] Force count mismatch in {out_path.name}: "
                          f"got {len(forces)}, expected {len(atoms)}")
            
            # Attach metadata
            atoms.info["REF_energy"] = energy_eV
            atoms.info["al_round"]   = round_num
            atoms.info["source"]     = "cp2k_sp"
            
            # --- Stress ---
            stress = parse_stress_from_out(out_content)
            if stress is not None:
                voigt = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
                atoms.info["REF_stress"] = voigt
            else:
                atoms.info["stress_missing"] = True

            if not forces_ok:
                atoms.info["forces_missing"] = True
            atoms.calc = None
            
            force_mags = np.linalg.norm(forces, axis=1)
    
            max_f = np.max(force_mags)
            # Accept
            new_frames.append(atoms)
            pool_hashes.add(geom_hash)
            atoms.info["_cp2k_dir"] = cp2k_dir
            print(f"  [+] {stem}  "
                  f"E={energy_eV:.4f} eV  "
                  f"residual={residual:.3f} eV/atom  "
                  f"forces={f'ok max={max_f:.3f}' if forces_ok else 'MISSING'}  "
                  f"stress={f'ok' if 'REF_stress' in atoms.info else 'MISSING'}  "
                  f"natoms={len(atoms)}")

        # Per-round summary (helps diagnose why frames were skipped)
        n_accepted = len(new_frames)  # cumulative; approximate for round summary
        print(f"  [→] Round {round_num} summary: "
              f"symlinks={n_symlinks}  incomplete={n_incomplete}  "
              f"excluded={n_excluded}  duplicates={n_dup}  "
              f"too_large={n_too_large}  parse_failed={parse_failed}  "
              f"new_this_round≈{len(new_frames)}")

    # ----------------------------------------------------------------
    # Step 4 — Write results
    # ----------------------------------------------------------------
    round_dirs = sorted(Path(p) for p in glob.glob("cp2k_sp_round*") if Path(p).is_dir())
    if not round_dirs:
        print("\n[!] No cp2k_sp_round* directories found.")
        return
    
    out_files = [f for rd in round_dirs for f in rd.rglob("*.out")]
    n_out      = len(out_files)
    n_in_pool  = len(pool)
    n_new      = len(new_frames)
    coverage   = (n_in_pool / n_out * 100) if n_out > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"  File coverage")
    print(f"{'='*60}")
    print(f"  .out files found        : {n_out}")
    print(f"  Frames in pool (before) : {n_in_pool}")
    print(f"  Coverage                : {coverage:.1f}%")

    print(f"\n{'='*60}")
    print(f"  Scan complete")
    print(f"{'='*60}")
    print(f"  New unique frames added : {n_new}")
    print(f"  Already in pool         : {already_have}")
    print(f"  Parse failures / bad    : {parse_failed}")
    print(f"  Written to              : {POOL_FILE}")
    print(f"  Max atoms allowed       : {MAX_ATOMS}")

    if not new_frames:
        print("\n[✓] Pool is already up to date — nothing to add.")
        return

    pool.extend(new_frames)
    write(POOL_FILE, pool, format="extxyz")

    n_final   = len(pool)
    coverage_final = (n_final / n_out * 100) if n_out > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Pool summary")
    print(f"{'='*60}")
    print(f"  Frames before           : {n_in_pool}")
    print(f"  Frames after            : {n_final}")
    print(f"  Coverage (after)        : {coverage_final:.1f}%")
    print(f"  →  {POOL_FILE}")

    # Update global geometry index
    geom_index = _load_geometry_index(CP2K_DIR)
    for atoms in new_frames:
        h = get_atoms_hash(atoms)
        geom_index[h] = {
            "name":     stem,                                  
            "cp2k_dir": atoms.info.pop("_cp2k_dir", CP2K_DIR)  
        }
    _save_geometry_index(CP2K_DIR, geom_index)
    print(f"[✓] Global geometry index updated")
 
def _write_outputs(new_frames, failed):
    print("\n[→] Writing per-system training files...")
    build_per_system_training_files(new_frames)

    # Load existing pool
    if os.path.exists(POOL_FILE):
        pool = read(POOL_FILE, index=":")
    else:
        pool = []

    # Deduplicate before appending — prevents duplicates if --parse is rerun
    existing_hashes = {get_atoms_hash(a) for a in pool}
    unique_new = [a for a in new_frames if get_atoms_hash(a) not in existing_hashes]
    skipped = len(new_frames) - len(unique_new)
    if skipped:
        print(f"[!] Skipped {skipped} duplicate frames already in pool")

    pool.extend(unique_new)
    write(POOL_FILE, pool, format="extxyz")

    archive = f"al_cp2k_results_round{ROUND}.extxyz"
    write(archive, unique_new, format="extxyz")

    print(f"\n[✓] Master pool updated: {len(pool)} total frames "
          f"({len(unique_new)} new this round, {skipped} duplicates skipped)")
 
 
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="MACE Active Learning Pipeline")
    
    # This allows you to call: python active_pipeline.py round 1
    parser.add_argument("--parse",     action="store_true")
    parser.add_argument("--reparse",   action="store_true")
    parser.add_argument("--e0",        action="store_true")
    parser.add_argument("--parse-all", action="store_true", dest="parse_all")
    parser.add_argument("--target",    type=int, default=None, help="Target round number for --parse-all and --recover (optional)")
    parser.add_argument("--dissolve",  action="store_true")
    parser.add_argument("--recover",   action="store_true", help="Analyze missing frames, FPS prioritize, and write new inputs")
    parser.add_argument("--audit-jobs", nargs="+", default=[], help="List of specific job names to audit")
    parser.add_argument("--requeue", action="store_true", help="Append the audited jobs to a submission script if they failed")
    parser.add_argument("--runs",      type=int, default=100, help="How many cp2k runs are required")
    parser.add_argument("--model",     type=str, default="mace-mp-0b3-medium-float32.model", help="What model to validate with?")
    parser.add_argument(
    "--exclude",
    nargs="*",
    default=[],
    metavar="KEYWORD",
    help="Exclude systems whose system_type contains any of these strings (case-insensitive). "
         "e.g. --exclude DRY WET bulk_water"
)
    
    parser.add_argument("round_num", type=int, nargs="?", default=None, help="Round number (optional)")  
    
    args = parser.parse_args()
    
   # 1. Handle global parameters first (Unlinked from the action chain)
    if args.model:
        MODEL_PATH = args.model
    if args.exclude:
        EXCLUDE_SYSTEM_KEYWORDS.extend(args.exclude)
        print(f"[→] Excluding system keywords: {EXCLUDE_SYSTEM_KEYWORDS}")    
    if args.round_num is not None:
        apply_round(args.round_num)
    else:
        apply_round(ROUND) # Fallback to default global variable
    if args.e0:
        elements = ["H", "C", "O", "F", "S", "Pt"]
        if args.parse:
            parse_e0_results(E0_DIR, elements)
        else:
            jobs = generate_e0s_input(elements, E0_DIR)
            _write_submission_script(
                Path(E0_DIR)/"submit_e0.sh", jobs, E0_DIR, "E0s", len(elements), 0
            )
    elif args.recover:
        recover_and_prioritize_missing(target_round=args.target, n_runs=args.runs)    
    elif args.audit_jobs:
        audit_and_requeue_specific_jobs(
            target_keywords=args.audit_jobs, 
            target_round=args.target, 
            requeue=args.requeue
        )
    elif args.reparse:
        reparse_failed()          
    elif args.parse_all:
        # --parse-all 4  sets round_num=4; --parse-all --target 4 also works
        target = args.target if args.target is not None else args.round_num
        parse_all_cp2k_outputs(target_round=target)
    elif args.parse:
        parse_and_update()        
    else:
        # This triggers when NO specific action flag (--parse, --parse-all, etc.) is passed
        # This matches your Bash Step 2: running a standard optimization round iteration
        print(f"Starting standard execution for round context...")
        DISSOLVED = args.dissolve
        N_SELECT_TOTAL = args.runs
        run_round()