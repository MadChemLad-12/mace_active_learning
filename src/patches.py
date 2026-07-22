# patches.py
import numpy as np
import torch
from typing import Optional, Dict
from ase import Atoms
from torch import Tensor
import torch_dftd.torch_dftd3_calculator as _dftd3_mod

def apply_dftd3_cell_patch():
    if getattr(_dftd3_mod.TorchDFTD3Calculator, "_cell_patch_applied", False):
        return

    def _patched_preprocess_atoms(self, atoms: Atoms) -> Dict[str, Optional[Tensor]]:
        pos = torch.tensor(atoms.get_positions(), device=self.device, dtype=self.dtype)
        Z = torch.tensor(atoms.get_atomic_numbers(), device=self.device)
        if any(atoms.pbc):
            cell: Optional[Tensor] = torch.tensor(
                np.array(atoms.get_cell()), device=self.device, dtype=self.dtype
            )
        else:
            cell = None
        pbc = torch.tensor(atoms.pbc, device=self.device)
        edge_index, S = self._calc_edge_index(pos, cell, pbc)
        if cell is None:
            shift_pos = S
        else:
            shift_pos = torch.mm(S, cell.detach())
        input_dicts = dict(
            pos=pos, Z=Z, cell=cell, pbc=pbc, edge_index=edge_index, shift_pos=shift_pos
        )
        return input_dicts

    _dftd3_mod.TorchDFTD3Calculator._preprocess_atoms = _patched_preprocess_atoms
    _dftd3_mod.TorchDFTD3Calculator._cell_patch_applied = True