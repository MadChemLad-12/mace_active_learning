### Default values to enable parsing in VSCODE
from dataclasses import dataclass, field
from typing import Dict, List

class ActivePipelineConfig:
    # --- General ---
    round: int = 1                     # overridden by args.target if provided
    n_select_total: int = 300          # overridden by args.runs
    max_atoms: int = 580               # cap to avoid oversized GPU jobs
    reuse_existing_cp2k: bool = True   # skip inputs for frames w/ valid CP2K output
    exclude_system_keywords: List[str] = field(default_factory=list)

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