import ase.io
import numpy as np
from collections import defaultdict
import hashlib
# Build composition matrix and solve for best-fit E0s
from sklearn.linear_model import LinearRegression
import numpy as np
from pathlib import Path
import json
#Example Json {Atomic number: energy eV}
#E0s = {1: -12.6294, 6: -146.3745, 8: -431.6014, 
#       9: -656.5253, 16: -274.7039, 78: -3264.7049}

MAX_FORCE_REF = 25.0   # eV/Å
MAX_RMSE      = 600    # meV/Å
NON_PT_THRESH = 5.3

E0_JSON = "E0s.json"
try:
    with open(E0_JSON, "r") as file:
        E0s_ref = json.load(file)

except FileNotFoundError:
    # Fix 1: Handle a missing file
    print(f"Error: The file '{E0_JSON}' could not be found.")
    E0s_ref = {}
    
elements = E0s_ref.keys()
frames = ase.io.read("master_train_pool.xyz", ":")

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
    mace_path="mace-mp-0b3-medium-float32.model"

from mace.calculators import MACECalculator
calc = MACECalculator(
    model_paths=mace_path,
    device="cuda",
    default_dtype="float32"
)

for index, atoms in enumerate(unique_frames):
    symbols_list = atoms.get_chemical_symbols()
    symbols_set  = set(symbols_list)
    positions    = atoms.get_positions()
    pt_count     = symbols_list.count("Pt")
    stype        = atoms.info.get("system_type", "unknown")

    # ── 1. Non-Pt z-coordinate check ─────────────────────────────────────────
    is_not_pt = (np.array(symbols_list) != "Pt")
    non_pt_z  = positions[is_not_pt, 2]
    is_slab = pt_count > 3

    if is_slab:
        # Generic slab check: look for any non-Pt atom buried below the slab surface.
        # Customise slab_element and NON_PT_THRESH at the top of this file for other systems.
        slab_element = "Pt"  # Change this if your slab uses a different element
        is_not_slab  = (np.array(symbols_list) != slab_element)
        non_slab_z   = positions[is_not_slab, 2]
        if len(non_slab_z) > 0 and np.min(non_slab_z) < NON_PT_THRESH:
            bad.append(atoms)
            bad_info.append(f"[non_slab_z_too_low] index={index} {stype} "
                            f"min_z={np.min(non_slab_z):.2f} Å  slab_count={pt_count}")
            continue

    # ── 2. Cohesive energy check (system-aware) ───────────────────────────────
    e_total  = atoms.info["REF_energy"]
    e_ref    = sum(E0s_ref[z] for z in atoms.numbers)
    coh      = (e_total - e_ref) / len(atoms)

    if   "Pt" in symbols_set and pt_count > 3:          # Pt slab
        coh_lo, coh_hi = -8.0, -3.0
    elif "Pt" in symbols_set and pt_count <= 3:          # dissolved Pt
        coh_lo, coh_hi = -8.0, -1.0
    elif any(s in symbols_set for s in ("F", "S", "C")): # Nafion-containing
        coh_lo, coh_hi = -8.0, -1.0
    elif symbols_set <= {"H", "O"}:                      # bulk water
        coh_lo, coh_hi = -6.0, -1.0
    else:                                                 # fallback
        coh_lo, coh_hi = -8.0, -1.0

    if not (coh_lo < coh < coh_hi):
        bad.append(atoms)
        bad_info.append(f"[cohesive_energy] {stype} index={index} "
                        f"coh={coh:.2f} eV/atom (allowed {coh_lo} to {coh_hi})")
        continue

    # ── 3. Reference force check (strict for all systems) ────────────────────
    ref_f     = atoms.arrays["REF_forces"]
    max_f_ref = np.max(np.linalg.norm(ref_f, axis=1))

    if max_f_ref > 25:
        bad.append(atoms)
        bad_info.append(f"[ref_force_too_large] {stype} index={index} "
                        f"max_ref_F={max_f_ref:.2f} eV/Å")
        continue

    # ── 4. MACE force RMSE check (system-aware) ───────────────────────────────
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    mace_f   = atoms_copy.get_forces()
    rmse     = np.sqrt(np.mean((mace_f - ref_f)**2)) * 1000
    max_mace = np.max(np.linalg.norm(mace_f, axis=1))

    if   "Pt" in symbols_set and pt_count > 3:           # Pt slab
        rmse_thresh = 400
    elif symbols_set <= {"H", "O"}:                      # bulk water
        rmse_thresh = 800
    elif any(s in symbols_set for s in ("F", "S")):      # Nafion
        rmse_thresh = 800
    else:                                                 # dissolved Pt / fallback
        rmse_thresh = MAX_RMSE

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
ase.io.write("training_clean.xyz", good)
ase.io.write("training_bad.xyz",   bad)

print(f"The E0s of the clean data is")
generate_E0s(good)
print(f"Compare this to your current E0s")
print(f"{E0s_ref}")


