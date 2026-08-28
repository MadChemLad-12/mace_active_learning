#!/usr/bin/env python3
"""
select_frames.py

Select the most valuable frames from a candidate database to add to
master_train.xyz, using:

  1) Descriptor-based farthest-point sampling (FPS) for structural diversity
     relative to what's already in master_train.xyz.
  2) (Optional) Committee disagreement across multiple trained MACE models
     to rank by predictive uncertainty.

Usage
-----
# Diversity only (feature 2 off):
python select_frames.py --database candidates.xyz --master master_train.xyz \
    --n-select 200 --no-committee

# Diversity + committee disagreement:
python select_frames.py --database candidates.xyz --master master_train.xyz \
    --n-select 200 --committee models/v3.model models/v4.model models/v5.model \
    --device cuda

Notes
-----
- Descriptors use dscribe's SOAP. Install with: pip install dscribe --break-system-packages
- Committee models are loaded via mace.calculators.MACECalculator. Adjust
  MODEL_LOADER below if your checkpoints need different loading logic
  (e.g. multihead models with a specific head name).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write


# --------------------------------------------------------------------------
# Descriptor computation (feature 1: diversity / FPS)
# --------------------------------------------------------------------------

def compute_soap_descriptors(all_atoms, species, r_cut=6.0, n_max=8, l_max=6):
    """Average per-structure SOAP descriptor for each Atoms object."""
    from dscribe.descriptors import SOAP

    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="inner",  # one averaged descriptor per structure
    )
    descriptors = soap.create(all_atoms, n_jobs=1)
    return np.asarray(descriptors)


def greedy_fps(candidate_descs, reference_descs, n_select):
    """
    Greedy farthest-point sampling.

    Selects candidate indices that maximize the minimum distance to the
    reference set (master_train.xyz) union previously-selected candidates.
    Returns indices into candidate_descs, ordered by selection order
    (most valuable first).
    """
    n_candidates = candidate_descs.shape[0]
    if reference_descs.shape[0] > 0:
        # min distance from each candidate to the reference set
        min_dists = _min_dist_to_set(candidate_descs, reference_descs)
    else:
        min_dists = np.full(n_candidates, np.inf)

    selected = []
    selected_mask = np.zeros(n_candidates, dtype=bool)

    n_select = min(n_select, n_candidates)
    for _ in range(n_select):
        # pick the candidate currently farthest from reference+selected
        remaining = np.where(~selected_mask)[0]
        if remaining.size == 0:
            break
        best_local = remaining[np.argmax(min_dists[remaining])]
        selected.append(best_local)
        selected_mask[best_local] = True

        # update min_dists with distance to the newly selected point
        new_dists = np.linalg.norm(
            candidate_descs - candidate_descs[best_local], axis=1
        )
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[best_local] = -np.inf  # never reselect

    return selected


def _min_dist_to_set(query_descs, ref_descs, chunk=2000):
    """Memory-friendly min pairwise distance from each query to ref set."""
    n = query_descs.shape[0]
    out = np.empty(n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        # (chunk, n_ref)
        diff = query_descs[start:end, None, :] - ref_descs[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        out[start:end] = d.min(axis=1)
    return out


# --------------------------------------------------------------------------
# Committee disagreement (feature 2: uncertainty, togglable)
# --------------------------------------------------------------------------

def compute_committee_disagreement(all_atoms, model_paths, device="cpu"):
    """
    Run each committee model on each structure, return per-structure
    disagreement score = std across models of (energy per atom) combined
    with mean force-vector std across atoms.

    Returns array of shape (n_structures,) — higher = more disagreement.
    """
    from mace.calculators import MACECalculator

    calcs = [
        MACECalculator(model_paths=[p], device=device) for p in model_paths
    ]

    n_models = len(calcs)
    n_structs = len(all_atoms)
    energies_per_atom = np.zeros((n_models, n_structs))
    force_std_per_struct = np.zeros((n_models, n_structs))

    for m_idx, calc in enumerate(calcs):
        for s_idx, atoms in enumerate(all_atoms):
            a = atoms.copy()
            a.calc = calc
            e = a.get_potential_energy() / len(a)
            f = a.get_forces()
            energies_per_atom[m_idx, s_idx] = e
            force_std_per_struct[m_idx, s_idx] = np.linalg.norm(f, axis=1).mean()

    energy_disagreement = energies_per_atom.std(axis=0)
    force_disagreement = force_std_per_struct.std(axis=0)

    # normalize each to [0, 1] and combine (equal weight; adjust as needed)
    def _norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    combined = 0.5 * _norm(energy_disagreement) + 0.5 * _norm(force_disagreement)
    return combined


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", required=True, help="Path to candidate frames (multi-frame xyz)")
    p.add_argument("--master", required=True, help="Path to master_train.xyz")
    p.add_argument("--n-select", type=int, required=True, help="Number of frames to select")
    p.add_argument("--out", default="selected_frames.xyz", help="Output xyz path")

    p.add_argument("--r-cut", type=float, default=6.0)
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--l-max", type=int, default=6)

    p.add_argument("--committee", nargs="*", default=None,
                    help="Paths to committee model files (e.g. v3.model v4.model v5.model). "
                         "Omit or use --no-committee to disable.")
    p.add_argument("--no-committee", action="store_true",
                    help="Explicitly disable committee disagreement (feature 2), diversity-only mode.")
    p.add_argument("--committee-weight", type=float, default=0.5,
                    help="Blend weight for committee score vs FPS rank when --committee is used (0-1).")
    p.add_argument("--device", default="cpu", help="cpu or cuda, for committee MACE calculators")

    p.add_argument("--pool-factor", type=int, default=3,
                    help="When using committee, first FPS-select pool_factor * n_select candidates "
                         "for diversity, then re-rank that pool by committee disagreement and keep n_select.")

    args = p.parse_args()

    use_committee = bool(args.committee) and not args.no_committee
    if args.committee and args.no_committee:
        print("Note: --committee given but --no-committee also set; running diversity-only.", file=sys.stderr)

    print(f"Reading master training set: {args.master}")
    master_atoms = read(args.master, index=":")
    print(f"  {len(master_atoms)} frames")

    print(f"Reading candidate database: {args.database}")
    candidate_atoms = read(args.database, index=":")
    print(f"  {len(candidate_atoms)} frames")

    species = sorted({sym for atoms in (master_atoms + candidate_atoms) for sym in atoms.get_chemical_symbols()})
    print(f"Species: {species}")

    print("Computing SOAP descriptors (this may take a while for large sets)...")
    master_descs = compute_soap_descriptors(master_atoms, species, args.r_cut, args.n_max, args.l_max) if master_atoms else np.empty((0, 0))
    candidate_descs = compute_soap_descriptors(candidate_atoms, species, args.r_cut, args.n_max, args.l_max)

    if not use_committee:
        print(f"Diversity-only mode: selecting {args.n_select} frames via greedy FPS.")
        selected_idx = greedy_fps(candidate_descs, master_descs, args.n_select)
    else:
        pool_size = min(len(candidate_atoms), args.n_select * args.pool_factor)
        print(f"Stage 1 (FPS): reducing {len(candidate_atoms)} candidates to a diverse pool of {pool_size}.")
        pool_idx = greedy_fps(candidate_descs, master_descs, pool_size)
        pool_atoms = [candidate_atoms[i] for i in pool_idx]

        print(f"Stage 2 (committee disagreement): scoring pool with models {args.committee}")
        disagreement = compute_committee_disagreement(pool_atoms, args.committee, device=args.device)

        # rank pool by disagreement, descending (most uncertain = most valuable)
        order = np.argsort(-disagreement)
        n_keep = min(args.n_select, len(pool_idx))
        selected_local = order[:n_keep]
        selected_idx = [pool_idx[i] for i in selected_local]

        print(f"Selected {len(selected_idx)} frames from committee-ranked pool.")

    selected_atoms = [candidate_atoms[i] for i in selected_idx]
    write(args.out, selected_atoms)
    print(f"Wrote {len(selected_atoms)} frames to {args.out}")


if __name__ == "__main__":
    main()