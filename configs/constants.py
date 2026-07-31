"""
constants.py
Chemistry and infrastructure facts that do NOT change between active-learning
rounds. If you're tuning a value between Round 5 and Round 6, it does not
belong here — put it in the relevant round_configs/roundN_config.py instead.

This file should rarely change. When it does, it's because you added a new
element to the system or moved to a new cluster (new LIBDIR path).
"""

import os

# --- CP2K basis/pseudopotential library location ---
# Must be set via config.local.sh (sourced before running any pipeline script)
LIBDIR = os.environ.get("CP2K_LIBDIR")
if not LIBDIR:
    raise ValueError(
        "CP2K_LIBDIR environment variable is not set! "
        "Did you source config.local.sh?"
    )

# --- Atomic number mapping ---
Z_MAP = {"H": 1, "Li": 3, "C": 6, "O": 8, "F": 9, "P": 15, "S": 16, "Pt": 78}

# --- Elements treated as metals (affects e.g. slab-height triage logic) ---
METALS = {"Pt", "Li"}  # extend if other transition metals are added later

# --- Location of training files ---
CLEAN_TRAIN  = "training_clean.xyz"
BAD_TRAIN    = "training_bad.xyz"
MASTER_TRAIN = "master_train_pool.xyz"
HELD_OUT     = "held_out.xyz"

# --- CP2K basis set / pseudopotential per element: (basis, potential) ---
KIND_PARAMS = {
    "H":  ("DZVP-MOLOPT-SR-GTH-q1",  "GTH-PBE-q1"),
    "C":  ("DZVP-MOLOPT-SR-GTH-q4",  "GTH-PBE-q4"),
    "O":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "F":  ("DZVP-MOLOPT-SR-GTH-q7",  "GTH-PBE-q7"),
    "S":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "Pt": ("DZVP-MOLOPT-SR-GTH-q18", "GTH-PBE-q18"),
}

# --- Default simulation cell (a, b, c) in Angstrom, keyed by substring
# matched against the structure/system name ---
DEFAULT_CELLS = {
    "default":            (11.099, 9.612,  33.000),   # Pt slab systems
    "naf_naf":            (22.198, 19.224, 29.790),    # Nafion + Pt slab
    "pt_nafion":          (22.198, 19.224, 29.790),
    "bulk_nafion":        (11.099, 9.612,  21.220),    # Bulk dissolved systems
    "bulk_water_pt":      (11.099, 9.612,  21.220),
    "dissolvedoh_nafion": (11.099, 9.612,  33.000),    # Dissolved oxide/hydroxide
    "dissolvedo2_nafion": (11.099, 9.612,  33.000),
    "dissolvedo_nafion":  (11.099, 9.612,  33.000),
}

# --- E0 reference data file (isolated-atom energies, produced once per
# round via the E0 calculation step, but the filename convention is stable) ---
E0_JSON = "configs/E0s.json"
E0_CELL_SIZE = 20.0  # Angstrom, cubic box for isolated-atom E0 calculations

# --- External training-set filename tags that should never be pruned
# from the training set during cleaning, regardless of other rules ---
EXTERNAL_SYSTEM_TYPES = ("mptrj", "oc25", "reico")
