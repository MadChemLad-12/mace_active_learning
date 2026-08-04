### Default values to enable parsing in VSCODE
from dataclasses import dataclass, field
from typing import Dict, List

class ActivePipelineConfig:
    # --- General ---
    round: int = 1                     # overridden by args.target if provided
    n_select_total: int = 100          # overridden by args.runs
    max_atoms: int = 580               # cap to avoid oversized GPU jobs
    reuse_existing_cp2k: bool = True   # skip inputs for frames w/ valid CP2K output
    exclude_system_keywords: List[str] = field(default_factory=list)
    check_slab_z: bool = False         # Whether to check if atoms go below the Z_slab
    
    # --- MACE ---
    device: str = "cuda"
    dtype: str = "float32"
        
    # --- Pathology geometry triage before CP2K (catches exploded frames early) ---
    geoopt_trigger: bool = True
    geoopt_trigger_force: float = 20.0   # eV/Å -- flags "broken", not "uncertain"
    geoopt_max_steps: int = 30           # cheap cap, not a full anneal
    geoopt_fmax_target: float = 2.0      # eV/Å -- "no longer exploding" target
    apply_d3: bool = True                # MACE+D3 vs MACE-only for this round

    # --- External datasets ---
    external_datasets: bool = False
    external_sources: Dict[str, dict] = field(default_factory=lambda: {
        # "mptrj_pt": {
        #     "path": "training_data/mptrj-gga-ggapu/master_ranked_MPtrj_structures.extxyz",
        #     "n_samples": 30,   # only 35 frames total -- take most of them
        # },
        # "oc25_pt": {
        #     "path": "hugface_data/train/master_ranked_OC25_structures.extxyz",
        #     "n_samples": 90,   # 93 frames total -- sample ~30
        # },
    })

    # --- REICO (random imaginary-chemical box) sampling ---
    reico_sampling: bool = True
    reico_num: int = 50                # random boxes generated per round
    reico_min_atoms: int = 20
    reico_max_atoms: int = 60
    reico_vol_per_atom: float = 12.0   # Å³/atom, condensed-phase packing density
    reico_min_dist_scale: float = 0.6  # scales (r_cov_a + r_cov_b) per-pair min distance

    @property
    def e0_dir(self) -> str:
        # Depends on `round`, so this is a property rather than a plain field --
        # guarantees it's always in sync even if round is overridden post-init.
        return f"cp2k_e0_round{self.round}"

class NebGeoRunConfig:
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


def get_coh_bounds(symbols_set: set, pt_count: int) -> tuple:
    """
    SCF sanity check bounds on raw CP2K energy.
    cohesive = (E_total - sum(E0_ref)) / n_atoms, eV/atom
    """
    if pt_count > 3:                                          # Pt slab
        coh_lo, coh_hi = -20.0, 10.0
    elif pt_count > 0:                                        # dissolved Pt
        coh_lo, coh_hi = -20.0, 10.0
    elif "P" in symbols_set or "N" in symbols_set:
        coh_lo, coh_hi = -20.0, 7.0
    elif any(s in symbols_set for s in ("F", "S", "C")):       # Nafion-containing
        coh_lo, coh_hi = -20.0, 10.0
    elif symbols_set <= {"H", "O"}:                            # bulk water
        coh_lo, coh_hi = -20.0, 10.0
    else:                                                       # fallback
        coh_lo, coh_hi = -20.0, 5.0
    return coh_lo, coh_hi