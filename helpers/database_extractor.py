#!/usr/bin/env python3
"""
database_extractor.py  (select_frames.py, cached/chunked version)

Select the most valuable frames from a candidate database to add to
master_train.xyz, using:

  1) Descriptor-based farthest-point sampling (FPS) for structural diversity
     relative to what's already in the master set.
  2) (Optional) Committee disagreement across trained MACE models.

What's new in this version
--------------------------
- SOAP is computed in chunks (--chunk-size frames at a time, parallel over
  --n-jobs). Each finished chunk is saved to a cache dir, so a crash or
  Ctrl-C loses at most one chunk, and re-running resumes where it stopped.
- Descriptors of the candidate database are reused across runs (the cache is
  keyed on file path/size/mtime + SOAP parameters + stride). The master set is
  cached the same way and simply recomputed when the file changes.
- FPS no longer allocates a full (N x D) temporary per iteration; it uses a
  matvec with precomputed squared norms.
- Output is written progressively in batches of --batch-size frames, in
  selection order (most valuable first), optionally paced by --batch-interval
  seconds between batches.
- --out may be a directory (no .xyz suffix): one xyz per frame + manifest.csv
  + a combined selected_frames.xyz at the end.

Usage
-----
python database_extractor.py --database MB-pol-104607.xyz \
    --master master_train_pool.xyz --n-select 200 --no-committee \
    --out structures/ --stride 5 --n-jobs 16 --batch-size 20
"""

# Keep BLAS/OpenMP single-threaded: parallelism comes from joblib via --n-jobs.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Keep BLAS/OpenMP single-threaded: parallelism comes from joblib via --n-jobs.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write


# --------------------------------------------------------------------------
# Cached, chunked SOAP descriptors
# --------------------------------------------------------------------------

def _cache_key(path, species, r_cut, n_max, l_max, stride, chunk_size):
    st = os.stat(path)
    payload = json.dumps(
        [str(Path(path).resolve()), st.st_size, st.st_mtime_ns, list(species),
         r_cut, n_max, l_max, stride, chunk_size],
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def compute_soap_chunked(atoms_list, species, cache_dir, label,
                         r_cut, n_max, l_max, chunk_size=1000, n_jobs=1):
    """
    Per-structure averaged SOAP, computed chunk by chunk with on-disk caching.
    Returns a float32 array (n_structures, n_features).
    """
    from dscribe.descriptors import SOAP

    n = len(atoms_list)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    soap = SOAP(species=species, periodic=True, r_cut=r_cut,
                n_max=n_max, l_max=l_max, average="inner")

    n_chunks = math.ceil(n / chunk_size)
    t0 = time.time()
    computed = 0
    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, n)
        path = cache_dir / f"chunk_{c:05d}.npy"

        # reuse a valid cached chunk
        if path.exists():
            try:
                if np.load(path, mmap_mode="r").shape[0] == e - s:
                    continue
            except Exception:
                pass  # corrupt/partial -> recompute

        desc = np.asarray(soap.create(atoms_list[s:e], n_jobs=n_jobs), dtype=np.float32)
        tmp = cache_dir / f"chunk_{c:05d}.tmp.npy"
        np.save(tmp, desc)
        os.replace(tmp, path)  # atomic: no half-written chunks

        computed += 1
        elapsed = time.time() - t0
        remaining_chunks = sum(
            1 for k in range(c + 1, n_chunks)
            if not (cache_dir / f"chunk_{k:05d}.npy").exists()
        )
        eta = elapsed / computed * remaining_chunks
        print(f"  [{label}] chunk {c + 1}/{n_chunks} done "
              f"({e}/{n} frames) elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
              flush=True)

    # assemble into one preallocated float32 array
    first = np.load(cache_dir / "chunk_00000.npy", mmap_mode="r")
    out = np.empty((n, first.shape[1]), dtype=np.float32)
    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, n)
        out[s:e] = np.load(cache_dir / f"chunk_{c:05d}.npy")
    return out


# --------------------------------------------------------------------------
# Farthest-point sampling
# --------------------------------------------------------------------------

def _min_sqdist_to_set(query, ref, chunk=500, ref_chunk=2000):
    """Min squared distance from each query row to the ref set (chunked matmul)."""
    n = query.shape[0]
    out = np.full(n, np.inf, dtype=np.float32)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    for qs in range(0, n, chunk):
        qe = min(qs + chunk, n)
        qb = query[qs:qe]
        q_sq = np.einsum("ij,ij->i", qb, qb)
        bmin = np.full(qe - qs, np.inf, dtype=np.float32)
        for rs in range(0, ref.shape[0], ref_chunk):
            re_ = min(rs + ref_chunk, ref.shape[0])
            d2 = q_sq[:, None] + ref_sq[None, rs:re_] - 2.0 * (qb @ ref[rs:re_].T)
            np.maximum(d2, 0.0, out=d2)
            bmin = np.minimum(bmin, d2.min(axis=1))
        out[qs:qe] = bmin
    return out


def greedy_fps(cand, ref, n_select, on_select=None):
    """
    Greedy FPS: repeatedly pick the candidate farthest from (reference U selected).
    Returns candidate indices in selection order. `on_select(idx, dist)` is
    called as soon as each frame is chosen (used for progressive output).
    """
    n = cand.shape[0]
    cand = np.ascontiguousarray(cand, dtype=np.float32)
    has_ref = ref.shape[0] > 0
    if has_ref:
        ref = np.ascontiguousarray(ref, dtype=np.float32)

    # Centre both sets on the candidate mean: distances are unchanged, but the
    # |a|^2 + |b|^2 - 2ab expansion loses far less float32 precision.
    mu = cand.mean(axis=0, dtype=np.float64).astype(np.float32)
    cand = cand - mu
    if has_ref:
        ref = ref - mu

    cand_sq = np.einsum("ij,ij->i", cand, cand)
    min_d2 = _min_sqdist_to_set(cand, ref) if has_ref else np.full(n, np.inf, dtype=np.float32)

    selected = []
    for _ in range(min(n_select, n)):
        best = int(np.argmax(min_d2))
        if not np.isfinite(min_d2[best]) and min_d2[best] < 0:
            break  # everything already selected
        dist = float(np.sqrt(min_d2[best])) if np.isfinite(min_d2[best]) else float("inf")
        selected.append(best)
        if on_select is not None:
            on_select(best, dist)

        d2 = cand_sq - 2.0 * (cand @ cand[best]) + cand_sq[best]
        np.maximum(d2, 0.0, out=d2)
        np.minimum(min_d2, d2, out=min_d2)
        min_d2[best] = -np.inf  # never reselect
    return selected


# --------------------------------------------------------------------------
# Committee disagreement (optional)
# --------------------------------------------------------------------------

def compute_committee_disagreement(all_atoms, model_paths, device="cuda"):
    from mace.calculators import MACECalculator

    calcs = [MACECalculator(model_paths=[p], device=device, float_type="float32")
             for p in model_paths]
    n_models, n_structs = len(calcs), len(all_atoms)
    e_pa = np.zeros((n_models, n_structs))
    f_mag = np.zeros((n_models, n_structs))

    for m, calc in enumerate(calcs):
        for s, atoms in enumerate(all_atoms):
            a = atoms.copy()
            a.calc = calc
            e_pa[m, s] = a.get_potential_energy() / len(a)
            f_mag[m, s] = np.linalg.norm(a.get_forces(), axis=1).mean()

    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    return 0.5 * _norm(e_pa.std(axis=0)) + 0.5 * _norm(f_mag.std(axis=0))


# --------------------------------------------------------------------------
# Progressive batch output
# --------------------------------------------------------------------------

class BatchWriter:
    """
    Collects selected frames and writes them out `batch_size` at a time, in
    selection order. If `interval` > 0, waits that many seconds between
    batches (useful when something downstream picks files up as they appear).
    """

    def __init__(self, outdir, atoms_list, orig_index, batch_size=20, interval=0.0):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.atoms_list = atoms_list      # indexable by local index
        self.orig_index = orig_index      # local index -> index in original database
        self.batch_size = batch_size
        self.interval = interval
        self.buffer = []                  # (local_idx, score)
        self.rank = 0
        self.batches = 0
        self.manifest = self.outdir / "manifest.csv"
        if not self.manifest.exists():
            self.manifest.write_text("rank,source_index,score,file\n")

    def add(self, local_idx, score):
        self.buffer.append((local_idx, score))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        if self.batches > 0 and self.interval > 0:
            print(f"  waiting {self.interval:.0f} s before next batch...", flush=True)
            time.sleep(self.interval)
        lines = []
        for local_idx, score in self.buffer:
            a = self.atoms_list[local_idx].copy()
            a.info["source_index"] = int(self.orig_index[local_idx])
            a.info["selection_rank"] = self.rank
            fname = f"frame_{self.rank:05d}.xyz"
            write(self.outdir / fname, a)
            lines.append(f"{self.rank},{int(self.orig_index[local_idx])},{score:.6g},{fname}\n")
            self.rank += 1
        with open(self.manifest, "a") as fh:
            fh.writelines(lines)
        self.batches += 1
        print(f"  wrote batch {self.batches}: frames up to rank {self.rank - 1} "
              f"-> {self.outdir}", flush=True)
        self.buffer = []


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--n-select", type=int, required=True)
    p.add_argument("--out", default="selected_frames.xyz",
                   help="Output .xyz file, or a directory (no .xyz suffix) for per-frame files.")
    p.add_argument("--split", action="store_true",
                   help="With a .xyz --out, also write one file per frame.")

    p.add_argument("--r-cut", type=float, default=6.0)
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--l-max", type=int, default=6)

    p.add_argument("--stride", type=int, default=1,
                   help="Only consider every Nth candidate frame (MD frames are highly correlated).")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Frames per SOAP chunk / cache file.")
    p.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1,
                   help="Parallel workers for SOAP (default: all cores).")
    p.add_argument("--cache-dir", default=".soap_cache")

    p.add_argument("--batch-size", type=int, default=20,
                   help="Write output in batches of this many frames (directory mode).")
    p.add_argument("--batch-interval", type=float, default=0.0,
                   help="Seconds to wait between batches (directory mode). 0 = no pause.")

    p.add_argument("--committee", nargs="*", default=None)
    p.add_argument("--no-committee", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pool-factor", type=int, default=3)

    args = p.parse_args()

    use_committee = bool(args.committee) and not args.no_committee
    if args.committee and args.no_committee:
        print("Note: --committee given but --no-committee also set; running diversity-only.",
              file=sys.stderr)

    print(f"Reading master training set: {args.master}")
    master_atoms = read(args.master, index=":")
    print(f"  {len(master_atoms)} frames")

    print(f"Reading candidate database: {args.database}")
    all_candidates = read(args.database, index=":")
    print(f"  {len(all_candidates)} frames")

    orig_index = np.arange(0, len(all_candidates), args.stride)
    candidate_atoms = [all_candidates[i] for i in orig_index]
    del all_candidates
    if args.stride > 1:
        print(f"  stride {args.stride}: using {len(candidate_atoms)} candidate frames")

    species = sorted(set().union(*(a.symbols.species() for a in master_atoms + candidate_atoms)))
    print(f"Species: {species}")

    soap_kw = dict(r_cut=args.r_cut, n_max=args.n_max, l_max=args.l_max,
                   chunk_size=args.chunk_size, n_jobs=args.n_jobs)
    root = Path(args.cache_dir)

    print("Computing SOAP descriptors (chunked, cached, resumable)...")
    mkey = _cache_key(args.master, species, args.r_cut, args.n_max, args.l_max, 1, args.chunk_size)
    master_descs = compute_soap_chunked(
        master_atoms, species, root / f"master_{Path(args.master).stem}_{mkey}", "master", **soap_kw)
    ckey = _cache_key(args.database, species, args.r_cut, args.n_max, args.l_max,
                      args.stride, args.chunk_size)
    candidate_descs = compute_soap_chunked(
        candidate_atoms, species, root / f"db_{Path(args.database).stem}_{ckey}", "database", **soap_kw)
    print(f"Descriptor matrix: {candidate_descs.shape}, "
          f"{candidate_descs.nbytes / 1e9:.2f} GB (float32)")

    dir_mode = not args.out.lower().endswith(".xyz")
    writer = (BatchWriter(args.out, candidate_atoms, orig_index,
                          args.batch_size, args.batch_interval) if dir_mode else None)

    if not use_committee:
        print(f"Diversity-only mode: selecting {args.n_select} frames via greedy FPS.")
        cb = (lambda i, d: writer.add(i, d)) if writer else None
        selected_idx = greedy_fps(candidate_descs, master_descs, args.n_select, on_select=cb)
    else:
        pool_size = min(len(candidate_atoms), args.n_select * args.pool_factor)
        print(f"Stage 1 (FPS): reducing {len(candidate_atoms)} candidates to a pool of {pool_size}.")
        pool_idx = greedy_fps(candidate_descs, master_descs, pool_size)
        pool_atoms = [candidate_atoms[i] for i in pool_idx]

        print(f"Stage 2 (committee disagreement): scoring pool with models {args.committee}")
        disagreement = compute_committee_disagreement(pool_atoms, args.committee, device=args.device)
        order = np.argsort(-disagreement)[:min(args.n_select, len(pool_idx))]
        selected_idx = [pool_idx[i] for i in order]
        if writer:
            for i in order:
                writer.add(pool_idx[i], float(disagreement[i]))

    if writer:
        writer.flush()

    selected_atoms = [candidate_atoms[i] for i in selected_idx]
    combined = Path(args.out) / "selected_frames.xyz" if dir_mode else Path(args.out)
    write(combined, selected_atoms)
    print(f"Wrote {len(selected_atoms)} frames to {combined}")

    if args.split and not dir_mode:
        for i, atoms in enumerate(selected_atoms):
            write(f"{args.out[:-4]}_{i}.xyz", [atoms])


if __name__ == "__main__":
    main()