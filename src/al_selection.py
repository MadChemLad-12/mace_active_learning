#!/usr/bin/env python3
"""
al_selection.py
===============
Drop-in replacement for the "score by MACE max force, take top-k" block in
active_pipeline.run_round().

What it does
------------
1. Cheap filters first: hash duplicates, already-computed geometries, atoms
   closer than 0.7 A.
2. Committee disagreement: every candidate is evaluated by N fine-tuned MACE
   models. Score = max over atoms of the per-atom force deviation
   (sqrt of the variance of the force vector across models, summed over x,y,z).
3. Forced TS selection: for each NEB path (file mace_neb_*.extxyz) the
   highest-energy interior image and its neighbours are always selected,
   regardless of their score. They skip the localized-force-outlier check
   and are protected from the pre-CP2K relaxation.
4. Trust window: keep frames with dev_lo <= dev_max <= dev_hi.
   Below lo: the models already agree (little to learn).
   Above hi: usually garbage geometry, where the models disagree wildly.
5. FPS on SOAP descriptors over the surviving pool so the batch is diverse.

Config attributes read with getattr (add them to ActivePipelineConfig, or
leave them out to use the defaults):
    committee_models   list[str]   REQUIRED, >= 2 fine-tuned model paths
    committee_head     str|None    same head for every model (multihead models)
    ts_neighbors       int = 1     images either side of the TS image
    dev_lo, dev_hi     float|None  eV/A; None -> 50th / 98th percentile of the batch
    fps_pool_factor    int = 3     FPS pool = factor * n slots to fill

D3 note: the committee models were trained on CP2K PBE-D3 labels, so D3 is
already inside them. D3 is NOT added here.
"""
import hashlib
from collections import defaultdict

import numpy as np
from ase.neighborlist import neighbor_list


# ------------------------------------------------------------------
# Calculators
# ------------------------------------------------------------------
def load_committee(config):
    """One MACECalculator per model path. No D3 (it is baked into the labels)."""
    from mace.calculators import MACECalculator

    from pathlib import Path
    paths = list(getattr(config, "committee_models", None) or [])
    if not paths:
        raise ValueError("Set config.committee_models to the model path(s) to use.")
    if not getattr(config, "use_committee", True):
        paths = paths[:1]          # single-model mode: first entry only
    elif len(paths) < 2:
        raise ValueError("use_committee=True needs at least 2 paths in committee_models.")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Model file(s) not found: {missing}")
    head = getattr(config, "committee_head", None)
    calcs = []
    for p in paths:
        kw = dict(model_paths=p, device=config.device, default_dtype=config.dtype)
        if head:
            kw["head"] = head   # needs a recent MACE version
        calcs.append(MACECalculator(**kw))
        print(f"  [committee] loaded {p}" + (f" (head={head})" if head else ""))
    return calcs


# ------------------------------------------------------------------
# Committee statistics for one frame
# ------------------------------------------------------------------
def committee_stats(atoms, calcs, topk=5):
    Fs, Es = [], []
    for c in calcs:
        a = atoms.copy()
        a.calc = c
        Fs.append(a.get_forces())
        Es.append(a.get_potential_energy())
    Fs, Es = np.array(Fs), np.array(Es)          # (M, N, 3), (M,)
    F_mean = Fs.mean(axis=0)
    # per-atom deviation: RMS over models of |F_m - F_mean|
    dev = np.sqrt(((Fs - F_mean) ** 2).sum(axis=-1).mean(axis=0))   # (N,)
    return {
        "dev_max": float(dev.max()),
        "dev_topk": float(np.sort(dev)[-topk:].mean()),
        "e_std_per_atom": float(Es.std() / len(atoms)),
        "e_mean": float(Es.mean()),
        "f_max": float(np.linalg.norm(F_mean, axis=1).max()),  # old score, kept for comparison
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def min_dist_ok(atoms, cutoff=0.7):
    _, _, d = neighbor_list("ijd", atoms, cutoff=cutoff)
    return len(d) == 0


def fps_indices(features, n):
    """Farthest-point sampling on row vectors; returns n indices."""
    n = min(n, len(features))
    if n <= 0:
        return []
    chosen = [int(np.argmax(np.linalg.norm(features - features.mean(0), axis=1)))]
    mind = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(features - features[nxt], axis=1))
    return chosen


def find_ts_indices(candidates, stat_fn, n_neighbors=1):
    """
    Group frames by (source file, system_type) for NEB files, assume file order
    is path order, and return indices of the highest-energy INTERIOR image
    (+/- n_neighbors) per path. Needs load_candidates to set atoms.info['_src'].
    The energy peak is MACE's estimate of the TS, which is why the neighbours
    are taken as well: DFT on 3 images will show if the peak is misplaced.
    """
    paths = defaultdict(list)
    for i, a in enumerate(candidates):
        src = a.info.get("_src", "")
        if src.startswith("mace_neb_"):
            paths[(src, a.info.get("system_type", ""))].append(i)

    ts = set()
    for key, idx in paths.items():
        if len(idx) < 3:
            continue
        E = np.array([stat_fn(i)["e_mean"] for i in idx])
        k = int(np.argmax(E[1:-1])) + 1              # interior only
        lo, hi = max(1, k - n_neighbors), min(len(idx) - 1, k + n_neighbors + 1)
        ts.update(idx[lo:hi])
        print(f"  [TS] {key[0]}: {len(idx)} images, peak at image {k} "
              f"(barrier ~ {E[k] - E[0]:.2f} eV vs first image), taking {hi - lo}")
    return ts


# ------------------------------------------------------------------
# Main selection
# ------------------------------------------------------------------
def select_by_committee(candidates, calcs, config, already_computed,
                        hash_fn, soap_fn, phys_check):
    """
    candidates       list[Atoms] from load_candidates (with info['_src'])
    calcs            from load_committee
    already_computed fn(atoms)->bool
    hash_fn          get_atoms_hash
    soap_fn          build_soap_features
    phys_check       fn(atoms, calc)->(ok, reason)  e.g.
                     lambda a, c: is_physically_reasonable(a, c, round_num=config.round)
    """
    n_select = config.n_select_total
    ts_nb = getattr(config, "ts_neighbors", 1)
    dev_lo = getattr(config, "dev_lo", None)          # absolute eV/A, optional
    dev_hi = getattr(config, "dev_hi", None)
    lo_pct = getattr(config, "dev_lo_pct", 50.0)
    hi_pct = getattr(config, "dev_hi_pct", 98.0)
    pool_factor = getattr(config, "fps_pool_factor", 3)

    cache = {}
    score_key = "dev_max" if len(calcs) > 1 else "f_max"
    print(f"[→] Scoring mode: {'committee disagreement' if score_key == 'dev_max' else 'single-model max force'}")
    def stat(i):
        if i not in cache:
            cache[i] = committee_stats(candidates[i], calcs)
        return cache[i]

    def eligible(i, seen):
        a = candidates[i]
        h = hash_fn(a)
        if h in seen or already_computed(a) or not min_dist_ok(a):
            return False
        seen.add(h)
        return True

    seen = set()

    # --- forced TS frames (energies come from ALL images of each path) ---
    print(f"\n[→] Committee ({len(calcs)} models): locating NEB transition states...")
    ts_all = find_ts_indices(candidates, stat, ts_nb)
    ts_idx = sorted(i for i in ts_all if eligible(i, seen))
    print(f"  [TS] {len(ts_idx)} TS-region frames forced into the selection "
          f"({len(ts_all) - len(ts_idx)} already computed/duplicate)")

    # --- score everything else ---
    rest = [i for i in range(len(candidates)) if i not in set(ts_idx) and eligible(i, seen)]
    print(f"[→] Scoring {len(rest)} remaining candidates with the committee...")
    ok_rest = []
    for n, i in enumerate(rest, 1):
        try:
            stat(i)
            ok_rest.append(i)
        except Exception as e:
            print(f"    [!] committee evaluation failed for frame {i}: {e}")
        if n % 100 == 0:
            print(f"    {n}/{len(rest)}")
    rest = ok_rest
    if not rest:
        print("  [!] no candidates left to score")

    dev = np.array([cache[i]["dev_max"] for i in rest]) if rest else np.array([0.0])
    pct = np.percentile(dev, [10, 50, 90, 98, 100])
    print(f"  dev_max percentiles (10/50/90/98/100): "
          + "  ".join(f"{p:.3f}" for p in pct) + " eV/A")
    lo = dev_lo if dev_lo is not None else float(np.percentile(dev, lo_pct))
    hi = dev_hi if dev_hi is not None else float(np.percentile(dev, hi_pct))
    print(f"  trust window: {lo:.3f} <= dev_max <= {hi:.3f} eV/A")

    in_window = sorted((i for i in rest if lo <= cache[i]["dev_max"] <= hi),
                       key=lambda i: -cache[i]["dev_max"])
    n_rest = max(0, n_select - len(ts_idx))
    pool = in_window[: pool_factor * n_rest]

    # --- physical sanity on the pool ---
    pool_ok = []
    for i in pool:
        ok, why = phys_check(candidates[i], calcs[0])
        if ok:
            pool_ok.append(i)
        else:
            print(f"  [✗] {candidates[i].info.get('system_type', '?')}: {why}")

    # --- diversity ---
    if len(pool_ok) > n_rest:
        feats = soap_fn([candidates[i] for i in pool_ok])
        pool_ok = [pool_ok[k] for k in fps_indices(feats, n_rest)]

    # --- assemble ---
    out = []
    ts_set = set(ts_idx)
    for i in ts_idx + pool_ok:
        a = candidates[i].copy()
        a.calc = None
        s = cache[i]
        a.info["committee_dev"] = s["dev_max"]
        a.info["committee_dev_topk"] = s["dev_topk"]
        a.info["committee_e_std"] = s["e_std_per_atom"]
        a.info["committee_fmax"] = s["f_max"]
        a.info["ts_forced"] = i in ts_set
        out.append(a)

    print(f"\n[✓] Selected {len(out)} frames: {len(ts_idx)} forced TS + {len(pool_ok)} "
          f"committee/FPS (window pool {len(in_window)})")
    return out


# ------------------------------------------------------------------
# Calibration: does committee disagreement actually predict error?
# Run after CP2K labels are in the pool. Compares against the old max-force score.
# ------------------------------------------------------------------
def _hash(atoms, precision=4):
    pos = np.round(atoms.get_positions(), precision).tobytes()
    return hashlib.md5(pos + atoms.get_atomic_numbers().tobytes()).hexdigest()


def calibrate(selected_xyz, pool_xyz, calc):
    """
    selected_xyz : al_selected_roundN.xyz (has committee_dev / committee_fmax)
    pool_xyz     : master pool with REF_forces
    calc         : one committee member (the model that did the scoring)
    Prints Spearman correlation of each score with the true per-frame force RMSE.
    Frames are matched by geometry hash; a few may be missed by rounding.
    """
    from ase.io import read
    from scipy.stats import spearmanr

    sel = {_hash(a): (a.info["committee_dev"], a.info["committee_fmax"])
           for a in read(selected_xyz, index=":") if "committee_dev" in a.info}
    dev, fmax, err = [], [], []
    for a in read(pool_xyz, index=":"):
        h = _hash(a)
        if h in sel and "REF_forces" in a.arrays:
            b = a.copy()
            b.calc = calc
            err.append(np.sqrt(((b.get_forces() - a.arrays["REF_forces"]) ** 2).mean()))
            dev.append(sel[h][0])
            fmax.append(sel[h][1])
    print(f"matched {len(err)} frames")
    if len(err) > 5:
        print(f"Spearman(committee dev , true F RMSE) = {spearmanr(dev, err)[0]:.2f}")
        print(f"Spearman(MACE max force, true F RMSE) = {spearmanr(fmax, err)[0]:.2f}")
