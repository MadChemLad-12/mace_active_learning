"""
Config for neb_model_compare.py.
This script lives outside the active-learning pipeline (analysis/comparison
tool), so it isn't versioned per-round like the others -- it's more like
`constants.py` in that it changes rarely, only when you add a new model to
compare or adjust validation logic.
"""

import os
from dataclasses import dataclass, field
from typing import Dict


# --- Models being compared ---
MODELS = {
    "foundational": {
        "path":  os.environ.get("MACE_FOUNDATION_MODEL", "mace-mp-0b3-medium-float32.model"),
        "label": "MACE-MP-0b3",
        "color": "#2196F3",
    },
    "finetuned": {
        "path":  os.environ.get("MACE_FINETUNED_MODEL"),
        "label": "Fine-tune",
        "color": "#F44336",
    },
}


@dataclass
class NebModelCompareConfig:
    device: str = "cuda"
    dtype: str = "float32"

    # --- Endpoint pre-relaxation ---
    relax_endpoints: bool = True
    endpoint_fmax: float = 0.05   # eV/Å

    # --- Structure validation (dissolution mover detection) ---
    dissolving_threshold: float = 2.0    # Å -- beyond this, atom is a "mover"
    stationary_threshold: float = 0.8    # Å -- beyond this but not dissolving = warning
    abort_on_validation_fail: bool = False # Whether to abort NEB if the validator finds the wrong number of movers


    # --- NEB ---
    n_images: int = 10
    neb_fmax: float = 0.05
    neb_optimizer: str = "FIRE"
    neb_max_steps: int = 500
    climb: bool = False
    fix_by_height: bool = True
    fix_height_threshold: float = 2.7    # Å

    # --- Pathology detection ---
    pathology_energy_spike: float = 2.0   # eV, flag if E jumps > this vs neighbour
    pathology_force_fmax: float = 5.0     # eV/Å, flag if max force exceeds this
    pathology_atom_disp: float = 3.0      # Å, flag if any atom moved > this vs init
    pathology_energy_abs: float = 5.0     # eV above initial energy -> absolute flag
    
    # --- DFT Refinement ---
    run_dft_refine: bool = True       # Toggle off if you want to skip DFT completely
    dft_command: str = "mpiexec -n 6 vasp_std" # Command template for running DFT NEB refinement (expects {input_dir} and {output_dir} placeholders)
    dft_parms: dict = {
    'ibrion': 2,           # Conjugate Gradient relaxation for ionic updates
    'isif':   2,           # Relax ions only; keep the cell dimensions fixed
    'nsw':    30,          # Cheap limit: Max 30 ionic steps to "polish" the MACE path
    'ediffg': -0.05,       # Target force convergence (eV/Å)
    'prec':   'Accurate',
    'nelm':   150,
    'ediff':  1e-6,
    'nbands': 500,         # Remember to update based on total electrons + 500 empty bands
    'ismear': -1,          # Fermi-Dirac
    'sigma':  0.1,
    'imix':   4,           # Broyden mixing
    'amix':   0.1,
    'bmix':   1.0,
    'gga':    'PE',         # PBE
    'ivdw':   11,          # Grimme D3
    'lcharg': False,
    'lwave':  False,
    }
    


CONFIG = NebModelCompareConfig()

# --- Species -> expected mover count ---
# Derived from chemical formula of the dissolving species (e.g. PtOH2 =
# 1 Pt + 1 O + 2 H = 4 atoms total dissolve together). Adjust if your
# structure-naming convention differs.
SPECIES_MOVERS: Dict[str, int] = {
    "Pt":    1,
    "PtO":   2,
    "PtOH":  3,
    "PtO2":  3,
    "PtOH2": 5,   # NOTE: kept as-is (matches your original file), but flagging:
                  # PtOH2 = 1 Pt + 1 O + 2 H = 4 atoms, not 5. The old comment
                  # said "O + O + H + H" which doesn't match the formula name.
                  # Worth double-checking which is correct -- I have not changed
                  # the value, only pointing out the mismatch.
}
