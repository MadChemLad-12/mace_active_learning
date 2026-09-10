#!/usr/bin/env python3
"""
generate_solvated_nafion.py

Pipeline to generate MACE active-learning training data: for each source
Nafion structure (simplified short-tail unit and full polymer), build
three 2-chain packing configurations, then solvate each at varying
hydration levels (lambda = H2O per -SO3H group).

Steps
-----
1. Read a single Nafion repeat unit, centre it, and align its long chain
   axis onto z (the canonical reference orientation).
2. Use packmol to place two copies of the canonical unit per
   configuration: "parallel" (side by side, same direction), "crossed"
   (one chain rotated 90 degrees onto x, forming an X with the other),
   and "antiparallel" (side by side, one flipped 180 degrees so the
   chains point in opposite directions).
3. Fill remaining space with water for each hydration value using packmol.
4. Run a short MACE-driven geometry optimization on each structure.
5. Write final structures (+ optional ASE-GUI view) for every
   source x configuration x hydration combination.

Requirements
------------
- ase
- mace-torch
- packmol binary on $PATH (https://m3g.github.io/packmol/)

Usage
-----
    python generate_solvated_nafion.py
    python generate_solvated_nafion.py --view          # pop open ASE GUI at the end
    python generate_solvated_nafion.py --skip-opt       # skip the MACE relaxation step
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE

# ---------------------------------------------------------------------------
# CONFIG -- edit these to match your setup
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("/home/user/Documents/Programs/Python/ASE/MACE/active_learning/solvated_naf")
HYDRATION_VALUES = [0, 9, 12, 15]            # lambda = n(H2O) / n(-SO3H)

# Both source structures get built into a dimer + solvated at every
# hydration value, each into its own output subfolder. Add/remove entries
# here to control which sets of training data get generated.
SOURCE_STRUCTURES = {
    "simple": Path("/home/user/Documents/Structures/Nafion/Simple_nafion.xyz"),
    "full": Path("/home/user/Documents/Structures/Nafion/Nafion_single_unit.pdb"),
}
CUBIC_BOX_SIZE = 10.0                       # Angstrom, cube side length (default)

# Override CUBIC_BOX_SIZE per source if needed -- e.g. the full polymer is
# much larger than the simplified short-tail unit and will likely need a
# bigger box for the dimer + solvation shell to fit without packmol failing.
BOX_SIZE_OVERRIDES = {
    "simple": 25.0,
    "full": 35.0,   # placeholder -- adjust based on the full polymer's extent
}

# MACE model used for the pre-relaxation. Point this at your latest
# active-learning checkpoint (e.g. Round 6 multihead model), or fall back
# to the mace-mp foundation model if you just want a rough clean-up.
MACE_MODEL_PATH = "/home/user/Documents/Programs/Python/ASE/MACE/active_learning/mace_V5_active_learning_stagetwo.model"
MACE_DEVICE = "cuda"                        # "cuda" or "cpu"
FMAX = 0.10                                 # eV/Angstrom, loose since this is pre-DFT relax
MAX_OPT_STEPS = 200

PACKMOL_BIN = "packmol"                     # must be on $PATH
PACKMOL_TOLERANCE = 2.0                     # Angstrom, standard packmol default

CONFIGURATIONS = ["parallel", "crossed", "antiparallel"]

# --- Atom indices you MUST check against your own Nafion_single_unit.pdb ---
# "Head" = the sulfonic-acid terminus (the -SO3H side chain end), used for
# head-to-head alignment. "Tail" = the opposite backbone terminus.
# Open the PDB in a viewer (ASE GUI / VMD) and confirm these indices.
SPECIES_INDEXES = {
    "simple": {
        "head": 0,      # S atom of -SO3H
        "tail": 15,     # terminal backbone carbon/fluorine
    },
    "full": {
        "head": 0,      # S atom of -SO3H
        "tail": 35,     # terminal backbone carbon/fluorine
    }
}

# ---------------------------------------------------------------------------
# STEP 1 + 2 : build the three 2-chain packing configurations
#
# Each source structure is first canonicalized (centred at the origin,
# long chain axis rotated onto z) using ASE. From that canonical single
# chain, packmol itself places and rotates two copies per configuration:
#
#   "parallel"      - both chains along z, side by side, same direction
#   "crossed"       - one chain along z, the other rotated onto x
#                      (crosses the first in an X)
#   "antiparallel"  - both chains along z, side by side, but one flipped
#                      180 degrees (head/tail reversed) -- parallel again,
#                      facing opposite directions
#
# This gives 3 base configurations per source structure, each of which
# then gets solvated across every hydration value in the next step.
# ---------------------------------------------------------------------------

def _principal_axis(atoms: Atoms) -> np.ndarray:
    """Return the unit vector of the largest-inertia principal axis
    (roughly the chain's long axis for an extended oligomer)."""
    positions = atoms.get_positions() - atoms.get_center_of_mass()
    masses = atoms.get_masses()
    inertia = np.zeros((3, 3))
    for m, r in zip(masses, positions):
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    eigval, eigvec = np.linalg.eigh(inertia)
    # smallest eigenvalue of the inertia tensor -> long (chain) axis
    return eigvec[:, np.argmin(eigval)]


def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        # 180 degree rotation: pick any orthogonal axis
        orthogonal = np.eye(3)[np.argmin(np.abs(a))]
        v = np.cross(a, orthogonal)
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def make_canonical_unit(source_path: Path, out_path: Path) -> Atoms:
    """Centre the source structure at the origin and rotate its long chain
    axis onto z. This is the reference orientation that packmol's per-copy
    rotation angles (below) are defined relative to."""
    atoms = read(source_path)
    atoms.translate(-atoms.get_center_of_mass())
    axis = _principal_axis(atoms)
    R = _rotation_matrix_from_vectors(axis, np.array([0.0, 0.0, 1.0]))
    atoms.positions = atoms.positions @ R.T

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(out_path, atoms)
    print(f"[1] Wrote canonical (z-aligned) unit -> {out_path}")
    return atoms


def _chain_configs(chain_span: float, box_size: float) -> dict:
    """Per-configuration (translation, rotation) pairs for the two chain
    copies, in packmol 'fixed x y z a b c' convention (angles in degrees,
    applied about x/y/z to the already-canonical -- z-aligned -- unit).
    Positions are absolute coordinates inside the box."""
    cx = cy = cz = box_size / 2
    sep = min(chain_span * 0.6, box_size * 0.3)  # side-by-side spacing, box-aware

    return {
        "parallel": [
            ((cx - sep / 2, cy, cz), (0, 0, 0)),
            ((cx + sep / 2, cy, cz), (0, 0, 0)),
        ],
        "crossed": [
            ((cx, cy, cz), (0, 0, 0)),        # chain along z
            ((cx, cy + 7.0, cz), (0, 90, 0)),       # rotated onto x -> crosses the first
        ],
        "antiparallel": [
            ((cx - sep / 2, cy, cz), (0, 0, 0)),
            ((cx + sep / 2, cy, cz), (180, 0, 0)),  # head/tail flipped
        ],
    }


PACKMOL_CONFIG_TEMPLATE = """\
tolerance {tolerance}
filetype pdb
output {output_pdb}

structure {unit_pdb}
  number 1
  fixed {x1} {y1} {z1} {a1} {b1} {c1}
end structure

structure {unit_pdb}
  number 1
  fixed {x2} {y2} {z2} {a2} {b2} {c2}
end structure
"""


def build_configuration(canonical_unit_path: Path, config_name: str,
                         box_size: float, work_dir: Path) -> Atoms:
    """Use packmol to place two copies of the canonical unit according to
    config_name ('parallel' | 'crossed' | 'antiparallel')."""
    unit_atoms = read(canonical_unit_path)
    chain_span = unit_atoms.positions[:, 2].max() - unit_atoms.positions[:, 2].min()

    placements = _chain_configs(chain_span, box_size)[config_name]
    (pos1, rot1), (pos2, rot2) = placements

    work_dir.mkdir(parents=True, exist_ok=True)
    packmol_inp = work_dir / f"packmol_{config_name}.inp"
    output_pdb = work_dir / f"{config_name}_dimer.pdb"

    packmol_inp.write_text(PACKMOL_CONFIG_TEMPLATE.format(
        tolerance=PACKMOL_TOLERANCE,
        output_pdb=output_pdb.name,
        unit_pdb=canonical_unit_path.resolve(),
        x1=pos1[0], y1=pos1[1], z1=pos1[2], a1=rot1[0], b1=rot1[1], c1=rot1[2],
        x2=pos2[0], y2=pos2[1], z2=pos2[2], a2=rot2[0], b2=rot2[1], c2=rot2[2],
    ))

    print(f"[2] Running packmol to build '{config_name}' configuration...")
    result = subprocess.run(
        [PACKMOL_BIN],
        stdin=open(packmol_inp, "r"),
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_pdb.exists():
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"packmol failed while building configuration '{config_name}'")

    dimer = read(output_pdb)
    dimer.set_cell([box_size] * 3)
    dimer.set_pbc(True)
    write(output_pdb, dimer)
    print(f"[2] Wrote '{config_name}' dimer -> {output_pdb}")
    return dimer


def count_sulfonic_groups(atoms: Atoms) -> int:
    """Count -SO3H groups by counting S atoms (assumes 1 S per sulfonic
    group and no other sulfur in the structure)."""
    n_s = sum(1 for s in atoms.get_chemical_symbols() if s == "S")
    if n_s == 0:
        raise ValueError(
            "No sulfur atoms found in the dimer -- can't infer -SO3H count. "
            "Check that Nafion_single_unit.pdb includes the sulfonic acid group."
        )
    return n_s


# ---------------------------------------------------------------------------
# STEP 3 : solvate with packmol at each hydration value
# ---------------------------------------------------------------------------

WATER_PDB_TEMPLATE = """\
HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.757   0.586   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.757   0.586   0.000  1.00  0.00           H
END
"""

PACKMOL_INPUT_TEMPLATE = """\
tolerance {tolerance}
filetype pdb
output {output_pdb}

structure {solute_pdb}
  number 1
  fixed {cx} {cy} {cz} 0. 0. 0.
  centerofmass
end structure

structure {water_pdb}
  number {n_water}
  inside cube 0. 0. 0. {box_size}
end structure
"""


def solvate_structure(dimer_path: Path, n_water: int, box_size: float,
                       work_dir: Path, tag: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    water_pdb = work_dir / "water_monomer.pdb"
    water_pdb.write_text(WATER_PDB_TEMPLATE)

    packmol_inp = work_dir / f"packmol_{tag}.inp"
    output_pdb = work_dir / f"nafion_hyd{tag}.pdb"

    cx = cy = cz = box_size / 2.0
    
    packmol_inp.write_text(PACKMOL_INPUT_TEMPLATE.format(
        tolerance=PACKMOL_TOLERANCE,
        output_pdb=output_pdb.name,
        solute_pdb=dimer_path.resolve(),
        water_pdb=water_pdb.name,
        n_water=n_water,
        box_size=box_size,
        cx=cx, cy=cy, cz=cz
    ))

    if n_water == 0:
        # packmol needs at least one structure block besides the solute in
        # some versions; for lambda=0 just copy the dry structure through.
        write(output_pdb, read(dimer_path))
        print(f"[3] lambda=0 -> copied dry structure -> {output_pdb}")
        return output_pdb

    print(f"[3] Running packmol for hydration={tag} ({n_water} waters)...")
    result = subprocess.run(
        [PACKMOL_BIN],
        stdin=open(packmol_inp, "r"),
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_pdb.exists():
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"packmol failed for hydration value {tag}")

    print(f"[3] Wrote solvated structure -> {output_pdb}")
    return output_pdb


# ---------------------------------------------------------------------------
# STEP 4 : MACE geometry optimization
# ---------------------------------------------------------------------------

def relax_with_mace(atoms: Atoms, model_path: str, device: str,
                     fmax: float, steps: int, traj_path: Path) -> Atoms:
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=model_path, device=device)
    atoms.calc = calc

    dyn = FIRE(atoms, trajectory=str(traj_path), logfile=str(traj_path.with_suffix(".log")))
    dyn.run(fmax=fmax, steps=steps)
    return atoms


# ---------------------------------------------------------------------------
# STEP 5 : write + optionally view final structures
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Open ASE GUI on final structures")
    parser.add_argument("--skip-opt", action="store_true", help="Skip the MACE relaxation step")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_structures = []

    for source_name, source_path in SOURCE_STRUCTURES.items():
        print(f"\n===== Source '{source_name}': {source_path} =====")

        source_out_dir = OUTPUT_DIR / source_name
        source_out_dir.mkdir(parents=True, exist_ok=True)
        box_size = BOX_SIZE_OVERRIDES.get(source_name, CUBIC_BOX_SIZE)

        # --- Step 1: canonicalize this source's single unit once ---
        canonical_path = source_out_dir / f"{source_name}_canonical_unit.pdb"
        make_canonical_unit(source_path, canonical_path)

        for config_name in CONFIGURATIONS:
            print(f"\n--- Configuration '{config_name}' ---")
            config_dir = source_out_dir / config_name

            # --- Step 2: pack the two chains via packmol for this configuration ---
            dimer_path = config_dir / f"{config_name}_dimer.pdb"
            dimer = build_configuration(canonical_path, config_name, box_size, config_dir)
            n_so3 = count_sulfonic_groups(dimer)
            print(f"[info] Detected {n_so3} -SO3H group(s) in the "
                  f"'{source_name}/{config_name}' dimer (box={box_size} A).")

            for lam in HYDRATION_VALUES:
                tag = str(lam)
                n_water = int(round(lam * n_so3))
                work_dir = config_dir / f"hydration_{tag}"

                # --- Step 3: solvate ---
                solvated_pdb = solvate_structure(dimer_path, n_water, box_size, work_dir, tag)
                atoms = read(solvated_pdb)
                atoms.set_cell([box_size] * 3)
                atoms.set_pbc(True)

                # --- Step 4: MACE pre-relaxation ---
                if not args.skip_opt:
                    traj_path = work_dir / f"relax_hyd{tag}.traj"
                    atoms = relax_with_mace(atoms, MACE_MODEL_PATH, MACE_DEVICE, FMAX, MAX_OPT_STEPS, traj_path)
                else:
                    print(f"[4] Skipping MACE relaxation for lambda={lam}")

                # --- Step 5: write final structure ---
                final_path = config_dir / f"{source_name}_{config_name}_hydration_{tag}.xyz"
                write(final_path, atoms)
                print(f"[5] Wrote final structure ({n_water} waters, lambda={lam}) -> {final_path}")
                final_structures.append(final_path)

    if args.view:
        from ase.visualize import view
        all_atoms = [read(p) for p in final_structures]
        view(all_atoms)


if __name__ == "__main__":
    main()