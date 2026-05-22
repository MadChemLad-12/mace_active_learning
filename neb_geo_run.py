"""
==============================================================================
MACE BATCH GEOMETRY OPTIMISATION — Pre-NEB Structure Screening
==============================================================================

WHAT THIS SCRIPT DOES (simply):

Before running expensive NEB calculations, we want to check that all our
input and final structures are physically sensible and at their lowest energy.

Think of it like this:
  - You have 8 different "starting positions" for your ball-rolling experiment
  - Before filming anything, you want to check each ball is actually sitting
    still in its valley, not wobbling or about to roll somewhere unexpected
  - This script checks all of them at once and gives you a report
  
  That's a lot of structures. Running NEB on an unstable starting geometry
  wastes hours of compute time. This script catches problems in minutes.

WHAT YOU GET:
  - Each structure geometry-optimised with MACE
  - A summary table showing which structures are stable
  - Energy differences between initial and final (reaction energies)
  - Optimised structures saved as new .cif files (ready for NEB input)
  - A JSON report with all energies
  - Tagged .extxyz files for EACH system, consumed by active_pipeline.py

HOW TO USE:
  1. Fill in the CONFIGURATIONS list below with your file paths
  2. Run:  python neb_geo_run.py
  3. Check the summary table printed at the end
  4. Run:  python active_pipeline.py  (reads the tagged .extxyz outputs)

==============================================================================
"""
from copy import deepcopy
import glob
import os
import json
import csv
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
from ase.mep import SingleCalculatorNEB
from ase.mep.neb import NEB
from scipy.optimize import linear_sum_assignment
from ase.geometry import find_mic
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# SETTINGS — CHANGE THESE
# ==============================================================================

MACE_MODEL_PATH = "mace-mp-0b3-medium-float32.model"
DEVICE          = "cuda"                     # "cuda" or "cpu"
DTYPE           = "float32"                  # Match your model's dtype
NODES           = 6                          # For parallel processing (if used)
OUTPUT_DIR      = "geo_opt_results"          # Where to save optimised structures
PATH_CSV        = "configs.csv"

# Geometry optimisation settings
FMAX            = 0.05     # Force convergence threshold (eV/Å)
MAX_STEPS       = 500      # Max optimisation steps per structure
OPTIMIZER       = "FIRE"   # "BFGS" (smooth surfaces) or "FIRE" (robust, any surface)
SKIP_OPTIMISATION = False   # Set to True to skip geometry optimisation

# Atom fixing
FIX_BY_HEIGHT         = False
FIX_HEIGHT_THRESHOLD  = 2.7    # Fix atoms below this Z (Angstrom)

# NEB Settings
SKIP_NEB      = False      # Set to True to skip NEB workflow
N_IMAGES      = 10         # Number of intermediate frames
NEB_FMAX      = 0.05       # Force threshold for the band
NEB_OPTIMIZER = "FIRE"     # FIRE is generally more stable for NEB
CLIMB         = True       # CI-NEB: finds the exact transition state
MAX_WARNINGS   = 8          # Max allowed atoms moved > FIX_HEIGHT_THRESHOLD before flagging
MAX_Threshold = 10.0         # Distance threshold (Å) for mapping consistency check

# ==============================================================================
# PLUMED / Metadynamics settings
# ==============================================================================
# PLUMED runs a short biased MD on the optimised INITIAL structure of each
# config.  It samples the local free-energy basin around the starting geometry,
# producing thermally-activated frames that NEB alone would never visit.
#
# WHEN TO USE:
#   SKIP_PLUMED = True   (default) — use for rounds 1-2 while the model is
#                        still learning basic chemistry.  Unbiased NEB is more
#                        stable with an immature potential.
#   SKIP_PLUMED = False  — enable from round 3 onwards once the model is
#                        reliable near the surface.  This is where PLUMED adds
#                        the most value: rare dissolution events, transition-
#                        state structures, solvation-shell rearrangements.
#
# CV STRATEGY (coordination number of surface Pt with O):
#   - CN_HIGH / CN_LOW define harmonic walls that keep the simulation near
#     the surface (local sampling mode, not full metadynamics).
#   - Increase PLUMED_STEPS and switch to METAD mode for actual free-energy
#     calculations in later rounds.
# ==============================================================================
# Logic: PLUMED is OFF by default. Typing --run-plumed makes SKIP_PLUMED = False
# PLUMED is a work in progress so do not expect good results
SKIP_PLUMED       = True
PLUMED_STEPS      = 5000
PLUMED_TEMP       = 300
PLUMED_DT         = 0.5      # Timestep in fs
PLUMED_FRICTION   = 0.02     # Langevin friction coefficient (fs⁻¹)
PLUMED_STRIDE     = 50       # Save frame + print CV every N steps
PLUMED_CN_R0      = 0.27     # Switching-function radius for Pt-O CN (nm)
PLUMED_CN_LOW     = 6.5      # Lower harmonic wall on CN (surface-like)
PLUMED_CN_HIGH    = 9.5      # Upper harmonic wall on CN (bulk-metal-like)
PLUMED_KAPPA      = 500.0    # Wall stiffness (kJ/mol)


# Active learning export settings
# The active_pipeline.py will look for these files.
# One file per system, named:  mace_geoopt_{name}.extxyz  (geo-opt frames)
#                              mace_neb_{name}.extxyz      (NEB frames)
AL_EXPORT_DIR = "al_candidates"  # sub-directory inside OUTPUT_DIR
AL_EXPORT_DIR = os.path.join(OUTPUT_DIR, "al_candidates")

# ==============================================================================
# CONFIGURATIONS — loaded from CSV
# ==============================================================================

with open(PATH_CSV, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    CONFIGURATIONS = [
        {
            "name":    row["Name"],
            "initial": row["initial"],
            "final":   row["final"]
        }
        for row in reader
    ]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def make_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, AL_EXPORT_DIR), exist_ok=True)
    print(f"[✓] Output directory: {OUTPUT_DIR}/")
    print(f"[✓] AL export directory: {OUTPUT_DIR}/{AL_EXPORT_DIR}/\n")

def check_mapping_consistency(atoms_ref, atoms_to_map, threshold=MAX_Threshold, max_warnings=MAX_WARNINGS):
    """
    Checks if any atoms in the mapped structure are physically too far 
    from their reference positions.
    """
    ref_symbols = atoms_ref.get_chemical_symbols()
    map_symbols = np.array(atoms_to_map.get_chemical_symbols())
    unique_elements = list(set(ref_symbols))
    cell = atoms_ref.get_cell()
    pbc  = atoms_ref.get_pbc()
    
    def check_element(element):
        """Find all atoms of one element that moved too far — runs in its own thread."""
        problems = []
        ref_indices = [i for i, s in enumerate(ref_symbols) if s == element]
        same_element_pos = atoms_to_map.positions[map_symbols == element]

        for i in ref_indices:
            diff, dists = find_mic(
                atoms_ref.positions[i] - same_element_pos,
                cell, pbc
            )
            if np.min(dists) > threshold:
                problems.append((i, element, np.min(dists)))

        return problems

    # Threshold Logic: Only trigger detailed logs if we cross the warning limit
    with ThreadPoolExecutor(max_workers=NODES) as executor:
        futures = [executor.submit(check_element, el) for el in unique_elements]
        problematic_atoms = [p for f in futures for p in f.result()]

    warning_counter = len(problematic_atoms)

    if warning_counter > max_warnings:
        print(f"\n[!] ALERT: Significant structural mismatch detected!")
        print(f"    {warning_counter} atoms moved more than {threshold} Å.")
        print(f"    This often indicates a Pt dissociation or a mapping error.")
        for idx, sym, d in problematic_atoms[:5]:
            print(f"    - Atom {idx} ({sym}): moved {d:.2f} Å")
        if len(problematic_atoms) > 5:
            print(f"    - ... and {len(problematic_atoms)-5} others.")
        return False

    return True

def map_atoms_by_proximity(atoms_ref, atoms_to_map, cutoff=8.0):
    """
    Pairs atoms between two structures by minimizing the total 
    displacement (accounting for Periodic Boundary Conditions).
    """
    # 1. Ensure chemical species match count-wise
    if atoms_ref.get_chemical_formula() != atoms_to_map.get_chemical_formula():
        raise ValueError("Chemical formulas do not match!")
    
    new_indices = np.zeros(len(atoms_ref), dtype=int)
    
    # Process element by element to ensure Oxygen doesn't map to Platinum
    symbols     = np.array(atoms_ref.get_chemical_symbols())
    map_symbols = np.array(atoms_to_map.get_chemical_symbols())
    unique_elements = list(set(symbols))
        
    def solve_element(element):
        """Solve the assignment problem for one element — runs in its own thread."""
        idx_ref = np.where(symbols     == element)[0]
        idx_map = np.where(map_symbols == element)[0]

        pos_ref = atoms_ref.positions[idx_ref]
        pos_map = atoms_to_map.positions[idx_map]

        tree       = cKDTree(pos_map)
        candidates = tree.query_ball_point(pos_ref, cutoff)

        cost_matrix = np.full((len(idx_ref), len(idx_map)), 1e6)

        for i, neighbor_indices in enumerate(candidates):
            if not neighbor_indices:
                continue
            diff, dists = find_mic(
            pos_ref[i] - pos_map[neighbor_indices],
            atoms_ref.get_cell(),
            atoms_ref.get_pbc()
                )
            for j, dist in zip(neighbor_indices, dists):
                cost_matrix[i, j] = dist
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return idx_ref[row_ind], idx_map[col_ind]
    
    # Run one thread per element — typically 3-5 elements (Pt, O, H, C, N...)
    with ThreadPoolExecutor(max_workers=NODES) as executor:
        futures = {element: executor.submit(solve_element, element)
                   for element in unique_elements}

        for element, future in futures.items():
            ref_indices, map_indices = future.result()
            new_indices[ref_indices] = map_indices

    return atoms_to_map[new_indices]

def load_mace():
    """Load MACE model once — reused for all structures."""
    print(f"[→] Loading MACE model from: {MACE_MODEL_PATH}")
    calc = MACECalculator(
        model_paths=MACE_MODEL_PATH,
        device=DEVICE,
        default_dtype=DTYPE
    )
    print(f"[✓] MACE model loaded\n")
    return calc


def get_fixed_indices(atoms):
    """Return list of atom indices to freeze, based on Z-height threshold."""
    fixed = []
    if FIX_BY_HEIGHT:
        for atom in atoms:
            if atom.position[2] < FIX_HEIGHT_THRESHOLD:
                fixed.append(atom.index)
    return fixed


def optimise_structure(atoms, calc, label, config_name):
    """
    Run a geometry optimisation on a single structure.
    Returns: (optimised_atoms, energy, converged, steps_taken, max_force)
    """
    atoms.calc = calc
    fixed_indices = get_fixed_indices(atoms)
    if fixed_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_indices))

    log_path  = os.path.join(OUTPUT_DIR, f"{config_name}_{label}.log")
    traj_path = os.path.join(OUTPUT_DIR, f"{config_name}_{label}_traj.traj")

    if OPTIMIZER == "FIRE":
        opt = FIRE(atoms, trajectory=traj_path, logfile=log_path)
    else:
        opt = BFGS(atoms, trajectory=traj_path, logfile=log_path)

    try:
        converged   = opt.run(fmax=FMAX, steps=MAX_STEPS)
        steps_taken = opt.get_number_of_steps()
        energy      = atoms.get_potential_energy()
        forces      = atoms.get_forces()

        if fixed_indices:
            free_mask = np.ones(len(atoms), dtype=bool)
            free_mask[fixed_indices] = False
            max_force = np.sqrt((forces[free_mask]**2).sum(axis=1)).max()
        else:
            max_force = np.sqrt((forces**2).sum(axis=1)).max()

        return atoms, energy, converged, steps_taken, max_force

    except Exception as e:
        print(f"    [✗] ERROR during optimisation of {config_name} {label}: {e}")
        return atoms, None, False, 0, None


def classify_stability(converged, max_force, steps_taken):
    """Return a simple stability verdict string."""
    if converged and max_force is not None and max_force < FMAX * 2:
        return "STABLE ✓"
    elif max_force is not None and max_force < 0.5:
        if steps_taken >= MAX_STEPS:
            return "NEARLY (~) [hit MAX_STEPS]"
        return "NEARLY (~)"
    else:
        return "UNSTABLE ✗"


def plot_energy():
    traj_files = glob.glob(os.path.join(OUTPUT_DIR, "*_traj.traj"))
    if not traj_files:
        print("[!] No trajectory files found.")
        return

    for traj_file in traj_files:
        try:
            traj     = read(traj_file, index=":")
            energies = [atoms.get_potential_energy() for atoms in traj]
            base     = os.path.basename(traj_file).replace("_traj.traj", "")

            if base.endswith("_initial"):
                state  = "initial"
                system = base.replace("_initial", "")
            elif base.endswith("_final"):
                state  = "final"
                system = base.replace("_final", "")
            else:
                state  = "unknown"
                system = base

            plt.figure(figsize=(8, 5))
            plt.plot(energies, marker='o', markersize=2)
            plt.xlabel("Optimization Step")
            plt.ylabel("Energy (eV)")
            plt.title(f"{system} ({state})")
            plt.tight_layout()

            out_file = os.path.join(OUTPUT_DIR, f"figures/{system}_{state}_energy.png")
            plt.savefig(out_file)
            plt.close()
            print(f"[✓] {out_file}")

        except Exception as e:
            print(f"[✗] Failed on {traj_file}: {e}")


def print_progress_bar(current, total, config_name):
    pct = int((current / total) * 40)
    bar = "█" * pct + "░" * (40 - pct)
    print(f"\n[{bar}] {current}/{total} — {config_name}")


# ==============================================================================
# NEB WORKFLOW  (fixed image-list construction + AL export)
# ==============================================================================

def neb_workflow(init_atoms, final_atoms, calc, name):
    """
    Run CI-NEB between init_atoms and final_atoms using MACE.
    Saves:
      - {OUTPUT_DIR}/{name}_neb.traj          (ASE trajectory)
      - {OUTPUT_DIR}/{name}_neb.extxyz        (all images, for visualisation)
      - {OUTPUT_DIR}/{AL_EXPORT_DIR}/mace_neb_{name}.extxyz
            (tagged for active_pipeline.py: system_type + neb_image set)
    """
    print(f"\n[→] Starting NEB for: {name}")

    # ---- FIX: correct image list construction ----
    # Previously:  [init.copy() for _ in range(N_IMAGES + 1) + [final.copy()]]
    # This crashed because you can't add a list to range() in Python 3.
    
    if len(init_atoms) != len(final_atoms):
        print(f"  [✗] ERROR: Initial and final structures have different number of atoms.")
        print(f"      Init: {len(init_atoms)} atoms, Final: {len(final_atoms)} atoms")
        return []
    elif init_atoms.get_chemical_symbols() != final_atoms.get_chemical_symbols():
        print(f"  [✗] ERROR: Initial and final structures have different chemical compositions.")
        print(f"      Init: {init_atoms.get_chemical_symbols()}")
        print(f"      Final: {final_atoms.get_chemical_symbols()}")
        return []
    else:
        for i, (s1, s2) in enumerate(zip(init_atoms.get_chemical_symbols(), final_atoms.get_chemical_symbols())):
            if s1 != s2:
                print(f"Index {i} is {s1} in initial but {s2} in final!")
                break
    
    
    try:
        final_atoms = map_atoms_by_proximity(init_atoms, final_atoms)
        print(f"  [✓] Atom mapping successful: final structure reordered to match initial.")
    except Exception as e:
        print(f"[✗] Mapping failed: {e}")
        
    if check_mapping_consistency(init_atoms,final_atoms):
        print(f"  [✓] Initial and final structures are consistent within {MAX_Threshold} Å with {MAX_WARNINGS} warnings.")
    else:
        print(f"  [!] WARNING: Initial and final structures show significant mismatch.")
        print(f"      This may lead to NEB failure or unphysical paths.")
        print(f"      Consider checking the structures visually or adjusting the mapping threshold.")
        
     
    images = [init_atoms.copy()]
    for _ in range(N_IMAGES):
        images.append(init_atoms.copy())
    images.append(final_atoms.copy())
    # -----------------------------------------------
    for image in images:
        image.calc = calc
    
    neb = NEB(images, climb=CLIMB, allow_shared_calculator=True)
    neb.interpolate(apply_constraint=False)  # Don't apply constraints to interpolated images

    fixed_indices = get_fixed_indices(init_atoms)
    if fixed_indices:
        for image in images:  
            image.set_constraint(FixAtoms(indices=fixed_indices))


    neb_traj_path = os.path.join(OUTPUT_DIR, f"{name}_neb.traj")
    neb_log_path  = os.path.join(OUTPUT_DIR, f"{name}_neb.log")

    if NEB_OPTIMIZER == "FIRE":
        optimizer = FIRE(neb, trajectory=neb_traj_path, logfile=neb_log_path)
    else:
        optimizer = BFGS(neb, trajectory=neb_traj_path, logfile=neb_log_path)

    try:
        t0 = time.perf_counter()
        optimizer.run(fmax=NEB_FMAX, steps=MAX_STEPS)
        print(f"[✓] NEB completed for {name} in {time.perf_counter() - t0:.2f} s")
    except Exception as e:
        print(f"[✗] NEB failed for {name}: {e}")

    # ---- Tag every image with metadata before export ----
    for idx, img in enumerate(images):
        img.info["system_type"] = name
        img.info["neb_image"]   = idx
        img.info["source"]      = "mace_neb"
        img.info["n_images"]    = len(images)

    # Standard visualisation output
    write(os.path.join(OUTPUT_DIR, f"{name}_neb.extxyz"), images, format="extxyz")

    # Active-learning candidate export (one file per system)
    al_path = os.path.join(OUTPUT_DIR, AL_EXPORT_DIR, f"mace_neb_{name}.extxyz")
    write(al_path, images, format="extxyz")
    print(f"[✓] NEB frames exported for AL: {al_path}")

    return images


# ==============================================================================
# GEO-OPT AL EXPORT HELPER
# ==============================================================================

def export_geoopt_for_al(name, init_atoms, final_atoms):
    """
    Write a single .extxyz for active_pipeline.py containing the
    optimised initial + final frames, tagged with system_type.
    """
    frames = []
    for label, atoms in [("initial", init_atoms), ("final", final_atoms)]:
        at = atoms.copy()
        at.info["system_type"] = name
        at.info["geoopt_label"] = label
        at.info["source"]       = "mace_geoopt"
        frames.append(at)

    al_path = os.path.join(OUTPUT_DIR, AL_EXPORT_DIR, f"mace_geoopt_{name}.extxyz")
    write(al_path, frames, format="extxyz")
    print(f"[✓] GeoOpt frames exported for AL: {al_path}")


# ==============================================================================
# PLUMBED WORKFLOW  (find unique configurations + export for AL)
# ==============================================================================
from ase.calculators.plumed import Plumed
from mace.calculators import MACECalculator
from ase.md.langevin import Langevin
from ase import units

def find_surface_pt_index(atoms):
    """
    Find the index of the most undercoordinated Pt atom — the one most
    likely to dissolve.  We define 'undercoordinated' as having the fewest
    Pt neighbours within 3.2 Å (roughly the first shell of bulk Pt at 2.77 Å).
 
    Why this matters: PLUMED needs a specific atom index to compute the
    coordination-number CV.  Hardcoding index 0 would be wrong for most
    structures.  Instead we detect it automatically so the function works
    for any slab geometry.
    """
    from ase.neighborlist import NeighborList, natural_cutoffs
 
    pt_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols())
                  if sym == "Pt"]
    if not pt_indices:
        return None
 
    cutoffs = [1.6] * len(atoms)   # 2 × 1.6 Å = 3.2 Å Pt–Pt cutoff
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
 
    min_cn  = 999
    best_pt = pt_indices[0]
    for idx in pt_indices:
        neighbours, _ = nl.get_neighbors(idx)
        pt_neighbours = sum(1 for n in neighbours
                            if atoms[n].symbol == "Pt")
        if pt_neighbours < min_cn:
            min_cn  = pt_neighbours
            best_pt = idx
 
    return best_pt   # 0-based ASE index

def build_plumed_input(atoms, name):
    """
    Generate a PLUMED input as a list of strings for the ASE Plumed wrapper.
 
    CV: coordination number of the most undercoordinated surface Pt with
        respect to all O atoms in the structure.
 
    Mode: harmonic walls (local basin sampling).
        This keeps the simulation near the starting geometry while still
        sampling thermal fluctuations — safe for an ML potential that has
        not yet seen dissolved configurations.
 
    PLUMED uses 1-based atom indices, so we add 1 to the ASE 0-based index.
    """
    pt_idx_0based = find_surface_pt_index(atoms)
    if pt_idx_0based is None:
        return None, None
 
    # Collect all O atom indices (1-based for PLUMED)
    o_indices_1based = [i + 1 for i, sym in enumerate(atoms.get_chemical_symbols())
                        if sym == "O"]
    if not o_indices_1based:
        print(f"  [!] No O atoms found in {name} — skipping PLUMED")
        return None, None
 
    pt_idx_1based  = pt_idx_0based + 1
    o_group        = ",".join(map(str, o_indices_1based))
    colvar_file    = os.path.join(OUTPUT_DIR, f"COLVAR_{name}")
 
    plumed_lines = [
        # Coordination number of the surface Pt with all O atoms
        f"cn: COORDINATION GROUPA={pt_idx_1based} GROUPB={o_group} "
        f"SWITCH={{RATIONAL R_0={PLUMED_CN_R0} NN=6 MM=12}}",
 
        # Harmonic walls — keep CN inside [CN_LOW, CN_HIGH]
        # This is local-basin sampling, NOT metadynamics.
        # The simulation explores thermal fluctuations but cannot dissolve.
        f"UPPER_WALLS ARG=cn AT={PLUMED_CN_HIGH} KAPPA={PLUMED_KAPPA}",
        f"LOWER_WALLS ARG=cn AT={PLUMED_CN_LOW}  KAPPA={PLUMED_KAPPA}",
 
        # Print CV trajectory for analysis
        f"PRINT ARG=cn FILE={colvar_file} STRIDE={PLUMED_STRIDE}",
    ]
 
    return plumed_lines, pt_idx_0based

def plumed_sampling(atoms, calc, name,
                    n_steps=None, timestep=None):
    """
    Run a short Langevin MD with MACE + PLUMED harmonic-wall bias to sample
    the local free-energy basin around the initial structure.
 
    Saves:
      - {OUTPUT_DIR}/{name}_plumed.extxyz        (all sampled frames)
      - {OUTPUT_DIR}/COLVAR_{name}               (CV trajectory for analysis)
      - {OUTPUT_DIR}/{AL_EXPORT_DIR}/mace_plumed_{name}.extxyz
            (tagged for active_pipeline.py)
 
    Returns a list of ASE Atoms (the saved frames), or [] on failure.
    """
    from ase.calculators.plumed import Plumed
    from ase.md.langevin import Langevin
    from ase import units
 
    if n_steps  is None: n_steps  = PLUMED_STEPS
    if timestep is None: timestep = PLUMED_DT
 
    print(f"\n[→] Starting PLUMED sampling for: {name}")
    print(f"    Steps: {n_steps} × {timestep} fs = {n_steps * timestep / 1000:.2f} ps")
 
    # Work on a copy so we don't mutate the NEB input
    sampling_atoms = atoms.copy()
 
    # Build the PLUMED input lines for this specific structure
    plumed_lines, pt_idx = build_plumed_input(sampling_atoms, name)
    if plumed_lines is None:
        print(f"  [!] Could not build PLUMED input for {name} — skipping")
        return []
 
    print(f"  [→] CV atom: Pt index {pt_idx} (0-based) — "
          f"most undercoordinated surface Pt")
 
    # Apply the same bottom-layer fix as geo-opt
    fixed_indices = get_fixed_indices(sampling_atoms)
    if fixed_indices:
        sampling_atoms.set_constraint(FixAtoms(indices=fixed_indices))
 
    # Wrap MACE with PLUMED.
    # The Plumed calculator intercepts positions at every MD step,
    # computes the CV, and adds bias forces on top of MACE forces.
    # timestep and kT must be passed so PLUMED can do unit conversions.
    try:
        plumed_calc = Plumed(
            calc=calc,                          # base MACE calculator
            input=plumed_lines,                 # CV + bias definitions
            timestep=timestep * units.fs,       # ASE internal units
            atoms=sampling_atoms,
            kT=units.kB * PLUMED_TEMP,
        )
    except Exception as e:
        print(f"  [✗] Failed to initialise Plumed calculator for {name}: {e}")
        return []
 
    sampling_atoms.calc = plumed_calc
 
    # Langevin thermostat — NVT ensemble at PLUMED_TEMP
    dyn = Langevin(
        sampling_atoms,
        timestep=timestep * units.fs,
        temperature_K=PLUMED_TEMP,
        friction=PLUMED_FRICTION,
    )
 
    # Collect frames in memory every PLUMED_STRIDE steps
    sampled_frames = []
 
    def _save_frame():
        at = sampling_atoms.copy()
        at.calc = None                          # strip live calculator
        at.info["system_type"]    = name
        at.info["source"]         = "mace_plumed"
        at.info["plumed_step"]    = dyn.get_number_of_steps()
        at.info["plumed_pt_idx"]  = int(pt_idx)
        sampled_frames.append(at)
 
    dyn.attach(_save_frame, interval=PLUMED_STRIDE)
 
    try:
        t0 = time.perf_counter()
        dyn.run(n_steps)
        elapsed = time.perf_counter() - t0
        print(f"  [✓] PLUMED sampling completed for {name} in {elapsed:.2f} s  "
              f"| {len(sampled_frames)} frames collected")
    except Exception as e:
        print(f"  [✗] PLUMED sampling failed for {name}: {e}")
        if not sampled_frames:
            return []
        print(f"  [~] Saving {len(sampled_frames)} frames collected before failure")
 
    if not sampled_frames:
        return []
 
    # Save full trajectory for visualisation
    traj_out = os.path.join(OUTPUT_DIR, f"{name}_plumed.extxyz")
    write(traj_out, sampled_frames, format="extxyz")
 
    # Export for active_pipeline.py
    al_path = os.path.join(OUTPUT_DIR, AL_EXPORT_DIR, f"mace_plumed_{name}.extxyz")
    write(al_path, sampled_frames, format="extxyz")
    print(f"  [✓] PLUMED frames exported for AL: {al_path}")
 
    return sampled_frames

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("  MACE BATCH GEOMETRY OPTIMISATION & NEB WORKFLOW")
    print("="*70)
    
    global MACE_MODEL_PATH
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",       type=int, default=1)
    parser.add_argument("--model",       type=str, default=None,
                        help="Override model path")
    parser.add_argument("--skip-plumed", action="store_true",
                        help="Skip PLUMED sampling (default for rounds 1-2)")
    parser.add_argument("--skip-neb", action="store_true",
                        help="Skip neb sampling")
    parser.add_argument("--run-plumed",  action="store_true",
                        help="Alias for --skip-plumed=False, enable PLUMED")
    args = parser.parse_args()
    
    if args.run_plumed:
        SKIP_PLUMED = False

    elif args.skip_plumed:
        SKIP_PLUMED = True

    else:
        # Default behaviour based on round number
        SKIP_PLUMED = (args.round <= 2)
    
    if args.skip_neb:
        SKIP_NEB = True
    else:
        SKIP_NEB = (args.round <= 1)
    
    if args.model:
        MACE_MODEL_PATH = args.model
        print(f"[✓] Model overridden by argument: {MACE_MODEL_PATH}")
    
    make_output_dir()
    calc  = load_mace()
    start = time.perf_counter()

    total_configs    = len(CONFIGURATIONS)
    optimisation_cache = {}
    results          = []
    print(f"The simulation will run on {DEVICE.upper()} with MACE model: {os.path.basename(MACE_MODEL_PATH)}")
    print(f"Program started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Starting calculations for {total_configs} configurations...\n")
    print(f"{'Config':<20} {'Initial':<25} {'Final':<25}")
    
    for i, config in enumerate(CONFIGURATIONS):
        print(f"Calculation {config['name']} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        name       = config["name"]
        init_path  = config["initial"]
        final_path = config["final"]

        print_progress_bar(i + 1, total_configs, name)
        print(f"  Initial: {init_path}")
        print(f"  Final:   {final_path}")

        config_result = {
            "name":         name,
            "initial_file": init_path,
            "final_file":   final_path,
            "neb_run":      False,
            "plumed_run": False,
        }

        # --- Optimise both endpoints ---
        for label, path in [("initial", init_path), ("final", final_path)]:
            if not os.path.exists(path):
                print(f"  [✗] File not found: {path}")
                config_result[f"{label}_status"] = "FILE NOT FOUND"
                continue

            atoms     = read(path)
            atoms.calc = calc

            if SKIP_OPTIMISATION:
                print(f"  [→] Skipping optimisation for {label}. Calculating energy...")
                e         = atoms.get_potential_energy()
                conv, steps, fmax = True, 0, 0.0
                status    = "SKIPPED (Used Raw)"
                opt_atoms = atoms
            else:
                if path in optimisation_cache:
                    print(f"  [→] Reusing cached results for {label} ({os.path.basename(path)})")
                    opt_atoms, e, conv, steps, fmax = optimisation_cache[path]
                else:
                    print(f"  [→] Optimising {label} structure...")
                    opt_atoms, e, conv, steps, fmax = optimise_structure(atoms, calc, label, name)
                    if e is not None:
                        optimisation_cache[path] = (opt_atoms.copy(), e, conv, steps, fmax)

            if e is not None:
                status = classify_stability(conv, fmax, steps)
                print(f"  [{'✓' if conv else '~'}] {label.capitalize()}: "
                      f"E = {e:.4f} eV | Steps = {steps} | {status}")

                out_file = os.path.join(OUTPUT_DIR, f"{name}_{label}_opt.cif")
                write(out_file, opt_atoms)

                config_result.update({
                    f"{label}_energy_eV": float(e),
                    f"{label}_converged": bool(conv),
                    f"{label}_fmax":      float(fmax),
                    f"{label}_steps":     int(steps),
                    f"{label}_status":    status,
                    f"{label}_opt_file":  out_file,
                })
            else:
                config_result[f"{label}_status"] = "FAILED"

        # --- Export geo-opt frames for active_pipeline.py ---
        init_cached  = optimisation_cache.get(init_path)
        final_cached = optimisation_cache.get(final_path)
        if init_cached and final_cached:
            export_geoopt_for_al(name, init_cached[0], final_cached[0])
        elif SKIP_OPTIMISATION:
            # Reconstruct from config_result if we skipped optimisation
            if os.path.exists(init_path) and os.path.exists(final_path):
                export_geoopt_for_al(name, read(init_path), read(final_path))

        # --- NEB ---
        init_ok  = ("STABLE" in config_result.get("initial_status", "")
                    or SKIP_OPTIMISATION)
        final_ok = ("STABLE" in config_result.get("final_status",   "")
                    or SKIP_OPTIMISATION)
 
        init_for_neb  = None
        final_for_neb = None
        if init_ok and os.path.exists(init_path):
            init_for_neb = (optimisation_cache[init_path][0]
                            if init_path in optimisation_cache
                            else read(init_path))
        if final_ok and os.path.exists(final_path):
            final_for_neb = (optimisation_cache[final_path][0]
                             if final_path in optimisation_cache
                             else read(final_path))
            
        if not SKIP_NEB:
            if init_for_neb is not None and final_for_neb is not None:
                neb_workflow(init_for_neb, final_for_neb, calc, name)
                config_result["neb_run"] = True
            else:
                print(f"  [!] Skipping NEB for {name}: endpoints not ready/stable.")
 
        # --- PLUMED local-basin sampling ---
        # Only runs on the INITIAL structure — we want to sample configurations
        # near the surface state, not the vacancy/dissolved state.
        # Skipped entirely if SKIP_PLUMED = True (recommended for rounds 1-2).

        if not SKIP_PLUMED:
            if init_for_neb is not None:
                print(f"  [→] Running PLUMED sampling for {name}...")
                print(f"      Sampling local minimum around initial structure")
                plumed_frames = plumed_sampling(init_for_neb, calc, name)
                config_result["plumed_run"] = len(plumed_frames) > 0
                config_result["plumed_frames"] = len(plumed_frames)
            else:
                print(f"  [!] Skipping PLUMED for {name}: initial structure not available.")
        else:
            print(f"  [→] Skipping PLUMED sampling for {name} (SKIP_PLUMED=True)")
            print(f"      Try in later rounds once the model is more reliable.")
            
            
        # --- Reaction energy ---
        if ("initial_energy_eV" in config_result
                and "final_energy_eV" in config_result):
            delta_e = (config_result["final_energy_eV"]
                       - config_result["initial_energy_eV"])
            config_result["reaction_energy_eV"]     = delta_e
            config_result["reaction_energy_kJ_mol"] = delta_e * 96.485
 
        results.append(config_result)
        print(f"  Completed {name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        
    # -----------------------------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------------------------
    end = time.perf_counter()
    print("\n\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"Total time: {end - start:.1f} s")
    print(f"\n{'Config':<20} {'Initial':<25} {'Final':<25} "
          f"{'ΔE (eV)':<12} {'NEB':<6} {'PLUMED'}")
    print("-"*70)

    ready_for_neb   = []
    needs_attention = []
    exclude_for_future_rounds = []  # Optional: track configs to exclude in future rounds due to severe instability or mapping issues

    for r in results:
        name    = r["name"]
        init_s  = r.get("initial_status", "?")
        final_s = r.get("final_status",   "?")
        delta_e = r.get("reaction_energy_eV", None)
        delta_str = f"{delta_e:+.4f}" if delta_e is not None else "N/A"
        neb_done   = "✓" if r.get("neb_run")    else "–"
        plumed_done = "✓" if r.get("plumed_run") else "–"
    

        has_error = ("FILE NOT FOUND" in init_s or "FAILED" in init_s or
                     "FILE NOT FOUND" in final_s or "FAILED" in final_s)

        if has_error:
            print(f"{name:<20} {init_s:<25} {final_s:<25} "
                  f"{'N/A':<12} {neb_done:<6} {plumed_done}  ✗ ERROR")
            needs_attention.append(name)
            continue


        both_stable = ("STABLE" in init_s and "STABLE" in final_s)
        neb_ready   = "✓ YES" if both_stable else "~ CHECK"
        print(f"{name:<20} {init_s:<25} {final_s:<25} {delta_str:<12} {neb_done:<6} {plumed_done}")

        if both_stable:
            ready_for_neb.append(name)
        else:
            needs_attention.append(name)

    print("="*70)
    print(f"\n[✓] Ready for NEB ({len(ready_for_neb)}/{len(CONFIGURATIONS)}):")
    for name in ready_for_neb:
        print(f"    • {name}")

    if needs_attention:
        print(f"\n[!] Need attention ({len(needs_attention)}/{len(CONFIGURATIONS)}):")
        for name in needs_attention:
            print(f"    • {name}")
        print(f"\n    Tips for unstable structures:")
        print(f"    - Increase MAX_STEPS (currently {MAX_STEPS})")
        print(f"    - Check structure visually in VESTA")
        print(f"    - Consider OPTIMIZER = 'FIRE' for difficult cases")

    # Save JSON report
    report_path = os.path.join(OUTPUT_DIR, "screening_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Full report saved to: {report_path}")
    print(f"[✓] AL candidate files in: {OUTPUT_DIR}/{AL_EXPORT_DIR}/")
    print(f"    → Run: python active_pipeline.py  to start the selection + CP2K step")
    print("\n" + "="*70 + "\n")

    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    plot_energy()


if __name__ == "__main__":
    main()
