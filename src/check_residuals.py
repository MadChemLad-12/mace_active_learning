import ase.io
import ase
import numpy as np
from collections import defaultdict
import hashlib
# Build composition matrix and solve for best-fit E0s
from sklearn.linear_model import LinearRegression
import numpy as np
from ase.config import cfg
import os
from pathlib import Path
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from patches import apply_dftd3_cell_patch
apply_dftd3_cell_patch()
import json
### CONSTANTS
import sys
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from configs.round_configs.round6_check_residual import CONFIG, get_residual_bounds, get_force_bounds
from configs.constants import EXTERNAL_SYSTEM_TYPES, E0_JSON, FOUNDATION_MODEL_PATH, MASTER_TRAIN, CLEAN_TRAIN, BAD_TRAIN

MAX_FORCE_REF = CONFIG.max_force_ref   # eV/Å
MAX_RMSE      = CONFIG.max_rmse    # meV/Å
NON_PT_THRESH = CONFIG.non_pt_thresh
MAX_COUNT     = CONFIG.max_count

try:
    with open(E0_JSON, "r") as file:
        E0s_ref = {int(k): v for k, v in json.load(file).items()}
        print(f"Your E0s are {E0s_ref}")

except FileNotFoundError:
    # Fix 1: Handle a missing file
    print(f"Error: The file '{E0_JSON}' could not be found.")
    E0s_ref = {}

if Path(MASTER_TRAIN).exists():
    frames = ase.io.read(MASTER_TRAIN, ":")
else:
    print(f"[!] {MASTER_TRAIN} not found.")

elements = E0s_ref.keys()

HASH_PRECISION = 4 
def get_atoms_hash(atoms):
    """Same function as in your pipeline — positions + atomic numbers MD5."""
    pos_data = np.round(atoms.get_positions(), HASH_PRECISION).tobytes()
    nuc_data = atoms.get_atomic_numbers().tobytes()
    return hashlib.md5(pos_data + nuc_data).hexdigest()

def find_residuals(frames):
    residuals = defaultdict(list)
    for atoms in frames:
        e_total = atoms.info["REF_energy"]
        e_ref = sum(E0s_ref[z] for z in atoms.numbers)
        e_residual_per_atom = (e_total - e_ref) / len(atoms)
        stype = atoms.info.get("system_type", "unknown")
        residuals[stype].append(e_residual_per_atom)

    for stype, vals in residuals.items():
        print(f"{stype}: mean residual = {np.mean(vals):.4f} eV/atom  "
            f"std = {np.std(vals):.4f} eV/atom")

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

# Deduplicate using the same MD5 logic as your pipeline
seen_hashes = {}
unique_frames = []
duplicates = []
duplicate_ids = []

for atoms in frames:
    h = get_atoms_hash(atoms)
    if h not in seen_hashes:
        seen_hashes[h] = atoms.info.get("system_type", "unknown")
        unique_frames.append(atoms)
    else:
        duplicates.append((atoms.info.get("system_type", "?"), h))
        duplicate_ids=atoms.info.get("ID")


print(f"Before dedup: {len(frames)}")
print(f"After dedup:  {len(unique_frames)}")
print(f"Duplicates removed: {len(duplicates)}")
print(f"Top ten duplicate Ids and their partners are here duplicate_ids")

good, bad, bad_info = [], [], []

model_pattern = "mace_V*_active_learning_final.model"
found_models = sorted(Path(".").glob(model_pattern))
if found_models and len(found_models)>3:
    print(f"Using model {found_models[-1]}")
    mace_path= found_models[-1]
else:
    print(f"No new model found using foundational")
    mace_path=FOUNDATION_MODEL_PATH

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
    
    # ── 0. Check it does not exceed atom count max ───────────────────────────
    if atom_number > MAX_COUNT:
        bad.append(atoms)
        bad_info.append(f"[too_many_atoms_in_structure] "
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

        # Get cell lengths (a, b, c)
        cell_lengths = atoms.cell.lengths()
        a_len, b_len = cell_lengths[0], cell_lengths[1]

        # Get Pt positions
        pt_pos = atoms.positions[is_pt]

        # Min/Max coordinates of Pt along X and Y
        min_x, max_x = np.min(pt_pos[:, 0]), np.max(pt_pos[:, 0])
        min_y, max_y = np.min(pt_pos[:, 1]), np.max(pt_pos[:, 1])

        # Check span along X and Y
        x_span = max_x - min_x
        y_span = max_y - min_y

        # A slab spans almost the entire cell width in X and Y (minus ~1 bond length)
        is_continuous_x = (x_span > (a_len - 2.5)) or (min_x < edge_margin and (a_len - max_x) < edge_margin)
        is_continuous_y = (y_span > (b_len - 2.5)) or (min_y < edge_margin and (b_len - max_y) < edge_margin)

        # A slab MUST be periodic along both X and Y
        return is_continuous_x and is_continuous_y

    is_external = any(tag in stype.lower() for tag in EXTERNAL_SYSTEM_TYPES)
    is_np = "nanoparticle" in stype.lower() or "cluster" in stype.lower()

    is_slab = (pt_count > 20) and not is_external and not is_np and is_pt_slab(atoms)

    if is_slab:
        is_not_pt = (np.array(symbols_list) != "Pt")
        non_pt_z  = positions[is_not_pt, 2]
        if len(non_pt_z) > 0 and np.min(non_pt_z) < NON_PT_THRESH:
            bad.append(atoms)
            bad_info.append(f"[non_pt_z_too_low] index={index} {stype} "
                            f"min_z={np.min(non_pt_z):.2f} Å  pt_count={pt_count}")
            continue

    # ── 2. Cohesive energy check (system-aware) ───────────────────────────────
    e_total  = atoms.info["REF_energy"]
    e_ref    = sum(E0s_ref[z] for z in atoms.numbers)
    coh      = (e_total - e_ref) / len(atoms)

    coh_lo, coh_hi = get_residual_bounds(symbols_set, pt_count)
    if not (coh_lo < coh < coh_hi):
        bad.append(atoms)
        bad_info.append(f"[cohesive_energy] {stype} index={index} "
                        f"coh={coh:.2f} eV/atom (allowed {coh_lo} to {coh_hi})")
        continue

    # ── 3. Reference force check (strict for all systems) ────────────────────
    ref_f     = atoms.arrays["REF_forces"]
    max_f_ref = np.max(np.linalg.norm(ref_f, axis=1))

    if max_f_ref > MAX_FORCE_REF:
        bad.append(atoms)
        bad_info.append(f"[ref_force_too_large] {stype} index={index} "
                        f"max_ref_F={max_f_ref:.2f} eV/Å")
        continue

    # ── 4. MACE force RMSE check (system-aware) ───────────────────────────────
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    mace_f   = atoms_copy.get_forces()
    if np.isnan(mace_f).any():
        bad.append(atoms)
        bad_info.append(f"[mace_nan_forces] {stype} index={index} MACE returned NaN forces.")
        continue
    rmse     = np.sqrt(np.mean((mace_f - ref_f)**2)) * 1000
    max_mace = np.max(np.linalg.norm(mace_f, axis=1))

    rmse_thresh = get_force_bounds(symbols_list, pt_count)
    if rmse > rmse_thresh:
        bad.append(atoms)
        bad_info.append(f"[high_rmse] {stype} index={index} "
                        f"RMSE={rmse:.1f} meV/Å (threshold={rmse_thresh}) "
                        f"max_MACE={max_mace:.2f}  max_REF={max_f_ref:.2f} eV/Å")
        continue

    # ── 5. Passed all checks ──────────────────────────────────────────────────
    good.append(atoms)

# Print summary
print(f"\nGood: {len(good)},  Bad: {len(bad)}")
print("\nRejected frames:")
for info in bad_info:
    print(f"  {info}")

# Write — both lists contain only Atoms objects now
ase.io.write(CLEAN_TRAIN, good)
ase.io.write(BAD_TRAIN,   bad)

print(f"The E0s of the clean data is")
generate_E0s(good)
print(f"Compare this to your current E0s")
print(f"{E0s_ref}")


