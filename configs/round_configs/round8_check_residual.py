"""
Round 6 config -- src/check_residual.py
Copy forward to round1_check_residual.py and diff against this one.

"""

from dataclasses import dataclass


@dataclass
class CheckResidualConfig:
    apply_d3: bool = False          # apply D3 correction to MACE energies
    max_force_ref: float = 20    # eV/Å, max force for a structure to be valid # 45 was the old value and 15 did kind well
    max_rmse: float = 510.0       # meV/Å, max RMSE force for a structure to be valid
    non_pt_thresh: float = 5.3     # Å, z-height threshold for non-Pt atoms entering slab
    max_count: int = 800           # max atom count allowed in a system
    
    device: str = "cuda"             # device for MACE calculations (cuda or cpu)
    dtype: str = "float32"           # dtype for MACE calculations (float32 or float64)
    
    # --- Residual validation bounds (SCF sanity check on raw CP2K energy) ---
    # cohesive = (E_total - sum(E0_ref)) / n_atoms, eV/atom
    # Pt count / element membership (see logic notes below).

    def get_residual_bounds(self, symbols_set: set, pt_count: int, config_type: str = "") -> tuple:
        """
        SCF sanity check bounds on raw CP2K energy.
        cohesive = (E_total - sum(E0_ref)) / n_atoms, eV/atom
        """
        if "reico_random" in config_type:
            coh_lo, coh_hi = -7.0, 7.0 
        elif pt_count > 3:                                          # Pt slab
            coh_lo, coh_hi = -10.0, 10.0
        elif pt_count > 0:                                        # dissolved Pt
            coh_lo, coh_hi = -10.0, 10.0
        elif "P" in symbols_set or "N" in symbols_set:
            coh_lo, coh_hi = -10.0, 10.0
        elif any(s in symbols_set for s in ("F", "S", "C")):       # Nafion-containing
            coh_lo, coh_hi = -10.0, 10.0
        elif symbols_set <= {"H", "O"}:                            # bulk water
            coh_lo, coh_hi = -10.0, 10.0
        else:                                                       # fallback
            coh_lo, coh_hi = -10.0, 5.0
        return coh_lo, coh_hi

    # --- MACE FORCE RMSE check (SCF sanity check on raw CP2K forces) ---
    # Pt count / element membership (see logic notes below).

    def get_force_bounds(self, symbols_set: set, pt_count: int, config_type: str = "") -> tuple:
        """
        SCF sanity check bounds on raw CP2K energy.
        cohesive = (E_total - sum(E0_ref)) / n_atoms, eV/atom
        """
        if   "reico_random" in config_type:                 # Reico samples
            rmse_thresh = 550        
        elif   "Pt" in symbols_set and pt_count > 3:           # Pt slab
            rmse_thresh = 550
        elif "P" in symbols_set or "N" in symbols_set:
            rmse_thresh = 550
        elif set(symbols_set) <= {"H", "O"}:                      # bulk water
            rmse_thresh = 350
        elif any(s in symbols_set for s in ("F", "S")):      # Nafion
            rmse_thresh = 550
        else:                                                 # dissolved Pt / fallback
            rmse_thresh = CONFIG.max_rmse
        return rmse_thresh


CONFIG = CheckResidualConfig()



