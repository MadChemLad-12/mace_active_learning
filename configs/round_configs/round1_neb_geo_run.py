"""
Round 6 config -- neb_geo_run.py
Copy forward to round7_neb_geo_run.py and diff against this one.
"""

from dataclasses import dataclass


@dataclass
class NebGeoRunConfig:
    round: int = 1
    # --- Device / precision ---
    device: str = "cuda"        # "cuda" or "cpu"
    dtype: str = "float32"      # must match your model's dtype
    nodes: int = 6              # see note on Singularity + multi-node below

    # --- Geometry optimisation ---
    fmax: float = 0.05                # eV/Å, force convergence threshold
    max_steps: int = 800              # max optimisation steps per structure
    optimizer: str = "FIRE"           # "BFGS" (smooth surfaces) or "FIRE" (robust)
    skip_optimisation: bool = False
    apply_d3: bool = True

    # --- Atom fixing ---
    fix_by_height: bool = False
    fix_height_threshold: float = 2.7   # Å, fix atoms below this z-height

    # --- NEB ---
    skip_neb: bool = False
    n_images: int = 10
    neb_fmax: float = 0.05
    neb_optimizer: str = "FIRE"
    climb: bool = False                  # CI-NEB: finds exact transition state
    max_warnings: int = 8                # max atoms > fix_height_threshold before flagging
    max_threshold: float = 10.0          # Å, distance threshold for mapping consistency check
    
    # --- AIMD (post-NEB sampling) ---
    skip_aimd: bool = True
    aimd_steps: int = 2000
    aimd_temp: float = 600.0            # K -- higher means more diverse sampling
    aimd_dt: float = 1.0                # fs, timestep (1.0 fs safe for most systems)
    aimd_friction: float = 0.01         # fs⁻¹, Langevin friction coefficient
    aimd_stride: int = 20               # save a frame every N steps
    aimd_target: str = "initial"        # "initial", "final", or "both"
    aimd_warmup: int = 200              # steps at low T before production
    aimd_warmup_temp: float = 100.0     # K


CONFIG = NebGeoRunConfig()

# NOTE on `nodes` + Singularity:
# Singularity itself doesn't manage multi-node parallelism -- that's still
# SLURM/MPI's job. `nodes` here should map to your `#SBATCH --nodes=` value,
# and your launch script should call `srun singularity exec --nv image.sif
# python neb_geo_run.py ...` so SLURM handles the multi-node orchestration
# and each rank runs inside its own container instance. Don't try to make
# the container itself aware of node count -- keep that at the SLURM layer.
