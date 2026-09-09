import ase.io
import ase
import numpy as np
from collections import defaultdict
import hashlib
from sklearn.linear_model import LinearRegression
from ase.config import cfg
import os
from pathlib import Path
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from patches import apply_dftd3_cell_patch
apply_dftd3_cell_patch()
import json
import importlib
### CONSTANTS
from configs.constants import E0_JSON, FOUNDATION_MODEL_PATH, MASTER_TRAIN, CLEAN_TRAIN, BAD_TRAIN, EXTERNAL_SYSTEM_TYPES
from configs.round_configs.schema import ActivePipelineConfig
from configs.round_configs.round1_check_residual import CheckResidualConfig  
ROUND = ActivePipelineConfig.round

def load_residual_config(ROUND: int) -> tuple[CheckResidualConfig, callable, callable]:
    if ROUND != 1:
        raise ValueError(
            f"Round '{ROUND}' is not supported. Please use round=1 for the initial check."
        )

    config_module = importlib.import_module(
        f"configs.round_configs.round{ROUND}_check_residual"
    )
    config = config_module.CONFIG
    return config, config.get_residual_bounds, config.get_force_bounds

CONFIG, get_residual_bounds, get_force_bounds = load_residual_config(ROUND)

MAX_FORCE_REF = CONFIG.max_force_ref   # eV/Å
MAX_RMSE      = CONFIG.max_rmse    # meV/Å
NON_PT_THRESH = CONFIG.non_pt_thresh
MAX_COUNT     = CONFIG.max_count

# ============================================================
# Hashing — must match active_pipeline.py exactly, or dedup
# and provenance will silently diverge between the two scripts.
# ============================================================
HASH_PRECISION = 4
MODEL_INFO_KEY = "model"   # same atoms.info key active_pipeline.py writes

def get_atoms_hash(atoms):
    """Geometry-only MD5 — same function as in the pipeline."""
    pos_data = np.round(atoms.get_positions(), HASH_PRECISION).tobytes()
    nuc_data = atoms.get_atomic_numbers().tobytes()
    return hashlib.md5(pos_data + nuc_data).hexdigest()

def get_labeled_hash(atoms, model_name):
    """
    Compound (geometry, model) fingerprint — same function as the pipeline.
    master.xyz intentionally stores the same geometry once per labeling
    model (verification runs), so dedup on geometry alone would silently
    drop all but one of those records.
    """
    geom_hash = get_atoms_hash(atoms)
    return hashlib.md5(f"{geom_hash}:{model_name}".encode()).hexdigest()

def get_labeled_hash_from_info(atoms):
    model_name = atoms.info.get(MODEL_INFO_KEY, "unknown")
    return get_labeled_hash(atoms, model_name)

try:
    with open(E0_JSON, "r") as file:
        E0s_ref = {int(k): v for k, v in json.load(file).items()}
        print(f"Your E0s are {E0s_ref}")
except FileNotFoundError:
    print(f"Error: The file '{E0_JSON}' could not be found.")
    E0s_ref = {}

if Path(MASTER_TRAIN).exists():
    frames = ase.io.read(MASTER_TRAIN, ":")
else:
    print(f"[!] {MASTER_TRAIN} not found.")
    frames = []

elements = list(E0s_ref.keys())

def find_residuals(frames):
    """Per-system-type residual summary, reusing the value the pipeline
    already stored at parse time when available (avoids recomputing)."""
    residuals = defaultdict(list)
    for atoms in frames:
        if "residual_eV_per_atom" in atoms.info:
            e_residual_per_atom = atoms.info["residual_eV_per_atom"]
        else:
            e_total = atoms.info["REF_energy"]
            e_ref = sum(E0s_ref[z] for z in atoms.numbers)
            e_residual_per_atom = (e_total - e_ref) / len(atoms)
        stype = atoms.info.get("system_type", "unknown")
        residuals[stype].append(e_residual_per_atom)

    print("\nResidual summary by system_type:")
    for stype, vals in residuals.items():
        print(f"  {stype}: mean residual = {np.mean(vals):.4f} eV/atom  "
              f"std = {np.std(vals):.4f} eV/atom  (n={len(vals)})")

def generate_E0s(frames):
    X = np.zeros((len(frames), len(elements)))
    y = np.zeros(len(frames))
    for i, atoms in enumerate(frames):
        for j, Z in enumerate(elements):
            X[i, j] = np.sum(atoms.numbers == Z)
        y[i] = atoms.info["REF_energy"]
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    fitted_E0s = dict(zip(elements, reg.coef_))
    print(fitted_E0s)
    return fitted_E0s

# ============================================================
# Deduplicate — on (geometry, model), matching active_pipeline.py.
# A plain geometry hash here would collapse the same structure labeled
# by two different models down to one, deleting verification data.
# ============================================================
seen_hashes   = {}
unique_frames = []
duplicates    = []                  # (system_type, hash, frame_id) for every dup found
duplicate_ids = defaultdict(list)   # hash -> list of frame IDs sharing it

for atoms in frames:
    h = get_labeled_hash_from_info(atoms)
    frame_id = atoms.info.get("ID", "?")
    if h not in seen_hashes:
        seen_hashes[h] = frame_id
        unique_frames.append(atoms)
    else:
        duplicates.append((atoms.info.get("system_type", "?"), h, frame_id))
        duplicate_ids[h].append(frame_id)

print(f"Before dedup: {len(frames)}")
print(f"After dedup:  {len(unique_frames)}")
print(f"Duplicates removed: {len(duplicates)}")
pool_changed = len(duplicates) > 0   # dedup alone changes what's on disk
if duplicate_ids:
    print("Top 10 duplicate (geometry, model) groups and their frame IDs:")
    for h, ids in list(duplicate_ids.items())[:10]:
        print(f"  {h[:10]}...  IDs: {[seen_hashes[h]] + ids}")

find_residuals(unique_frames)

good, bad, bad_info = [], [], []
# pool_changed already set above (True if dedup dropped anything); per-frame
# loop below flips it True further if any frame falls through the cache.

# ============================================================
# Choose the MACE model used for THIS verification pass, and label
# every output frame with it — distinct from atoms.info["model"], which
# records whichever model originally selected/labeled the frame during
# active learning. This is "what checked it," not "what picked it."
# ============================================================
model_pattern = "mace_V*_active_learning_final.model"
found_models = sorted(Path(".").glob(model_pattern))
if found_models and len(found_models) > 3:
    print(f"Using model {found_models[-1]}")
    mace_path = found_models[-1]
else:
    print(f"No new model found, using foundational")
    mace_path = FOUNDATION_MODEL_PATH

VERIFICATION_MODEL_NAME = Path(mace_path).stem

from mace.calculators import MACECalculator
calc_mace = MACECalculator(
    model_paths=mace_path,
    device=CONFIG.device,
    default_dtype=CONFIG.dtype
)
if CONFIG.apply_d3:
    print(f"[→] Including D3 in calculations (MACE + D3)")
    calc_DFT = TorchDFTD3Calculator(
                    device=CONFIG.device,
                    damping="bj",
                    xc=cfg.get("dispersion_xc", "pbe"),
                    cutoff=cfg.get("dispersion_cutoff", 40.0),
                )
    calc = SumCalculator([calc_mace, calc_DFT])
else:
    print(f"[→] Using MACE only (no D3)")
    calc = calc_mace

for index, atoms in enumerate(unique_frames):
    symbols_list = atoms.get_chemical_symbols()
    symbols_set  = set(symbols_list)
    positions    = atoms.get_positions()
    pt_count     = symbols_list.count("Pt")
    stype        = atoms.info.get("system_type", "unknown")
    atom_number  = len(atoms)
    labeling_model = atoms.info.get(MODEL_INFO_KEY, "unknown")

    # ── Cache check: has THIS model already verified this frame? ─────────────
    # verification_model (not `model`, which is selection provenance) is the
    # field tied to the calculator identity. If it matches the model we just
    # loaded, the stored verdict is still valid — REF_* never changes, and
    # residual/fmax are REF-derived so they can't have drifted either. Only
    # the MACE-vs-REF comparison (step 4) actually depends on which model is
    # loaded, so a match means: skip everything, reuse the verdict.
    cached_model = atoms.info.get("verification_model")
    if cached_model == VERIFICATION_MODEL_NAME and "curation_status" in atoms.info:
        if atoms.info["curation_status"] == "good":
            good.append(atoms)
        else:
            bad.append(atoms)
            bad_info.append(f"[cached] {stype} index={index} model={labeling_model} "
                            f"reason={atoms.info.get('curation_reason', 'unknown')}")
        continue

    # Model changed (or frame never verified before) — recompute below and
    # overwrite verification_model/curation_status with fresh results.
    # REF_energy / REF_forces / REF_stress are CP2K ground truth and are
    # never touched here.
    pool_changed = True
    atoms.info["verification_model"] = VERIFICATION_MODEL_NAME

    # ── 0. Check it does not exceed atom count max ───────────────────────────
    if atom_number > MAX_COUNT:
        bad.append(atoms)
        atoms.info["curation_status"] = "bad"
        atoms.info["curation_reason"] = "too_many_atoms_in_structure"
        bad_info.append(f"[too_many_atoms_in_structure] model={labeling_model} "
                        f"system_count={atom_number}, MAX_COUNT={MAX_COUNT}")
        continue

    # ── 1. Non-Pt z-coordinate check ─────────────────────────────────────────
    is_not_pt = (np.array(symbols_list) != "Pt")
    non_pt_z  = positions[is_not_pt, 2]

    def is_pt_slab(atoms, pt_thresh=20, edge_margin=1.5):
        """
        Determines if a system containing Pt is a periodic slab vs an isolated nanoparticle.

        Parameters:
        -----------
        atoms : ase.Atoms
        pt_thresh : int
            Minimum number of Pt atoms to consider as a slab candidate.
        edge_margin : float (Angstroms)
            If Pt atoms get closer than this margin to BOTH cell edges along X or Y,
            it is forming periodic bonds across the boundary (i.e., a slab).
        """
        symbols = np.array(atoms.get_chemical_symbols())
        is_pt = (symbols == "Pt")
        pt_count = np.sum(is_pt)

        if pt_count < pt_thresh:
            return False

        cell_lengths = atoms.cell.lengths()
        a_len, b_len = cell_lengths[0], cell_lengths[1]

        pt_pos = atoms.positions[is_pt]

        min_x, max_x = np.min(pt_pos[:, 0]), np.max(pt_pos[:, 0])
        min_y, max_y = np.min(pt_pos[:, 1]), np.max(pt_pos[:, 1])

        x_span = max_x - min_x
        y_span = max_y - min_y

        is_continuous_x = (x_span > (a_len - 2.5)) or (min_x < edge_margin and (a_len - max_x) < edge_margin)
        is_continuous_y = (y_span > (b_len - 2.5)) or (min_y < edge_margin and (b_len - max_y) < edge_margin)

        return is_continuous_x and is_continuous_y

    is_external = any(tag in stype.lower() for tag in EXTERNAL_SYSTEM_TYPES)
    is_np = "nanoparticle" in stype.lower() or "cluster" in stype.lower()

    is_slab = (pt_count > 20) and not is_external and not is_np and is_pt_slab(atoms)

    if is_slab:
        if len(non_pt_z) > 0 and np.min(non_pt_z) < NON_PT_THRESH:
            bad.append(atoms)
            atoms.info["curation_status"] = "bad"
            atoms.info["curation_reason"] = "non_pt_z_too_low"
            bad_info.append(f"[non_pt_z_too_low] index={index} {stype} model={labeling_model} "
                            f"min_z={np.min(non_pt_z):.2f} Å  pt_count={pt_count}")
            continue

    # ── 2. Cohesive energy check (system-aware) ───────────────────────────────
    # Reuse the residual the pipeline already computed at parse time when
    # present — same formula, avoids recomputing, and guarantees the two
    # scripts agree on the number even if E0_JSON is refit later.
    if "residual_eV_per_atom" in atoms.info:
        coh = atoms.info["residual_eV_per_atom"]
    else:
        e_total = atoms.info["REF_energy"]
        e_ref   = sum(E0s_ref[z] for z in atoms.numbers)
        coh     = (e_total - e_ref) / len(atoms)

    # NOTE: these bounds are this script's own final-curation thresholds
    # (get_residual_bounds), deliberately separate from the coarser
    # coh_ok flag the pipeline stamped at parse time (get_coh_bounds) —
    # do not treat coh_ok as a substitute for this check.
    coh_lo, coh_hi = get_residual_bounds(symbols_set, pt_count)
    if not (coh_lo < coh < coh_hi):
        bad.append(atoms)
        atoms.info["curation_status"] = "bad"
        atoms.info["curation_reason"] = "cohesive_energy"
        bad_info.append(f"[cohesive_energy] {stype} index={index} model={labeling_model} "
                        f"coh={coh:.2f} eV/atom (allowed {coh_lo} to {coh_hi})")
        continue

    # ── 3. Reference force check (strict for all systems) ────────────────────
    ref_f = atoms.arrays["REF_forces"]
    max_f_ref = atoms.info.get("fmax")
    if max_f_ref is None:
        max_f_ref = np.max(np.linalg.norm(ref_f, axis=1))

    if max_f_ref > MAX_FORCE_REF:
        bad.append(atoms)
        atoms.info["curation_status"] = "bad"
        atoms.info["curation_reason"] = "ref_force_too_large"
        bad_info.append(f"[ref_force_too_large] {stype} index={index} model={labeling_model} "
                        f"max_ref_F={max_f_ref:.2f} eV/Å")
        continue

    # ── 4. MACE force RMSE check (system-aware) ───────────────────────────────
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    mace_f = atoms_copy.get_forces()
    if np.isnan(mace_f).any():
        bad.append(atoms)
        atoms.info["curation_status"] = "bad"
        atoms.info["curation_reason"] = "mace_nan_forces"
        bad_info.append(f"[mace_nan_forces] {stype} index={index} model={labeling_model} "
                        f"MACE ({VERIFICATION_MODEL_NAME}) returned NaN forces.")
        continue
    rmse = np.sqrt(np.mean((mace_f - ref_f) ** 2)) * 1000
    max_mace = np.max(np.linalg.norm(mace_f, axis=1))

    rmse_thresh = get_force_bounds(symbols_list, pt_count)
    if rmse > rmse_thresh:
        bad.append(atoms)
        atoms.info["curation_status"] = "bad"
        atoms.info["curation_reason"] = "high_rmse"
        bad_info.append(f"[high_rmse] {stype} index={index} model={labeling_model} "
                        f"RMSE={rmse:.1f} meV/Å (threshold={rmse_thresh}) "
                        f"max_MACE={max_mace:.2f}  max_REF={max_f_ref:.2f} eV/Å")
        continue

    # ── 5. Passed all checks ──────────────────────────────────────────────────
    atoms.info["rmse_meV_A"] = float(rmse)
    atoms.info["curation_status"] = "good"
    atoms.info.pop("curation_reason", None)
    good.append(atoms)

# Print summary
print(f"\nGood: {len(good)},  Bad: {len(bad)}")
print("\nRejected frames:")
for info in bad_info:
    print(f"  {info}")

# Write — both lists contain only Atoms objects now
ase.io.write(CLEAN_TRAIN, good)
ase.io.write(BAD_TRAIN,   bad)

# Persist curation_status / verification_model / rmse_meV_A back to the
# master pool itself — otherwise every field set above (including the
# cache this script relies on) is recomputed and discarded on the next run.
# Only write if something actually changed this run (new/re-verified frames,
# or dedup dropped something) if every frame hit the cache, master_train_pool.xyz
# already has exactly what we'd write, so skip the I/O entirely.
if pool_changed:
    ase.io.write(MASTER_TRAIN, unique_frames, format="extxyz")
    print(f"[✓] Wrote curation results back to {MASTER_TRAIN} ({len(unique_frames)} frames)")
else:
    print(f"[✓] {MASTER_TRAIN} already up to date — no frames needed recomputation, skipped write")

print(f"\nThe E0s of the clean data is")
generate_E0s(good)
print(f"Compare this to your current E0s")
print(f"{E0s_ref}")