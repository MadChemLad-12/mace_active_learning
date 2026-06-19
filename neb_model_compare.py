"""
==============================================================================
MACE MULTI-MODEL NEB COMPARISON — Foundational vs Fine-Tuned
==============================================================================

PURPOSE:
  Run NEB simulations for Pt dissolution pathways (Pt, PtO, PtOH, PtO2,
  PtOH2) on solvated Pt(111) at varying O coverages using TWO MACE models
  simultaneously — a foundational model and a fine-tuned model — and compare
  their outputs systematically.

WORKFLOW:
  1. VALIDATE  — check that endpoints only differ by the expected dissolving
                 atoms (1 for Pt, 2 for PtO/PtOH2, 3 for PtOH, 3 for PtO2)
  2. NEB RUN   — run NEB with each model; save outputs to separate directories
  3. COMPARE   — energy barriers, TS positions, path agreement (RMSE), forces
  4. PATHOLOGY — flag per-image anomalies: energy spikes, force divergence,
                 atom displacement outliers

EXPECTED CONFIG NAMES (parsed automatically):
  Close{coverage}Pt, Close{coverage}PtO, Close{coverage}PtOH,
  Close{coverage}PtO2, Close{coverage}PtOH2

USAGE:
  python neb_model_compare.py [--csv PATH] [--validate-only] [--no-neb]

==============================================================================
"""

import os
import csv
import json
import time
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.mep.neb import NEB
from ase.mep import NEBTools
from mace.calculators import MACECalculator

# ==============================================================================
# SETTINGS — EDIT THESE
# ==============================================================================

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "foundational": {
        "path":  os.environ.get("MACE_FOUNDATION_MODEL", "mace-mp-0b3-medium-float32.model"),
        "label": "MACE-MP-0b3",
        "color": "#2196F3",
    },
    "finetuned": {
        "path":  os.environ.get("MACE_FINETUNED_MODEL", "mace_V4_active_learning_stagetwo.model"),
        "label": "MACE-V4",
        "color": "#F44336",
    },
}
PATH_CSV    = os.environ.get("MACE_NEB_CSV", "Pt_Diss_Neb_test.csv")
OUTPUT_ROOT = os.environ.get("MACE_NEB_OUTPUT", "neb_comparison")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE  = "float32"

# ── ENDPOINT PRE-RELAXATION OPTIONS ─────────────────────────────────────────
RELAX_ENDPOINTS = True        # Set to True to optimize initial/final structures first
ENDPOINT_FMAX = 0.05          # Maximum residual force tolerance (eV/Å)

# ── Structure validation ───────────────────────────────────────────────────────
# Maximum distance (Å) an atom must move to be counted as "dissolving"
DISSOLVING_THRESHOLD = 2.0      # atoms beyond this are flagged as movers
# Maximum distance an atom may move AND still be called "stationary"
STATIONARY_THRESHOLD = 0.8      # atoms beyond this but not dissolving → warning
# Whether to abort NEB if the validator finds the wrong number of movers
ABORT_ON_VALIDATION_FAIL = False

# ── NEB ───────────────────────────────────────────────────────────────────────
N_IMAGES      = 10
NEB_FMAX      = 0.05
NEB_OPTIMIZER = "FIRE"
NEB_MAX_STEPS = 500
CLIMB         = False
FIX_BY_HEIGHT = True
FIX_HEIGHT_THRESHOLD = 2.7      # Å — fix atoms below this z-height

# ── Pathology detection ───────────────────────────────────────────────────────
PATHOLOGY_ENERGY_SPIKE  = 2.0   # eV — flag image if E jumps > this vs neighbour
PATHOLOGY_FORCE_FMAX    = 5.0   # eV/Å — flag image if max force exceeds this
PATHOLOGY_ATOM_DISP     = 3.0   # Å — flag image if any atom moved > this vs init
PATHOLOGY_ENERGY_ABS    = 5.0   # eV above initial energy → absolute flag

# ==============================================================================
# SPECIES → EXPECTED MOVER COUNT
#   Derived from chemical formula of the dissolving species.
#   PtOH2 → Pt + O + H + H = 4 atoms dissolve? No — the SPECIES dissolves
#   together, so PtOH2 = 1 Pt + 1 O + 2 H = 4 atoms total.
#   Adjust these if your naming convention differs.
# ==============================================================================
SPECIES_MOVERS = {
    "Pt":    1,    # only the Pt atom dissolves
    "PtO":   2,    # Pt + O
    "PtOH":  3,    # Pt + O + H
    "PtO2":  3,    # Pt + O + O
    "PtOH2": 5,    # Pt + O + O + H + H
}

# ==============================================================================
# HELPERS
# ==============================================================================

def parse_species_from_name(name: str) -> str | None:
    """
    Extract the dissolving species from a config name like 'Close0.75PtOH'.
    Returns one of: Pt, PtO, PtOH, PtO2, PtOH2, or None if unrecognised.
    """
    for species in sorted(SPECIES_MOVERS.keys(), key=len, reverse=True):
        if species in name:
            return species
    return None


def get_fixed_indices(atoms):
    """Indices of atoms to freeze based on Z-height threshold."""
    if not FIX_BY_HEIGHT:
        return []
    return [a.index for a in atoms if a.position[2] < FIX_HEIGHT_THRESHOLD]


def mic_displacements(atoms_init, atoms_final):
    """
    Return per-atom displacement magnitudes (Å) between two structures,
    respecting minimum image convention for periodic boundary conditions.
    """
    cell = atoms_init.get_cell()
    pbc  = atoms_init.get_pbc()
    diffs, dists = find_mic(
        atoms_final.positions - atoms_init.positions,
        cell, pbc
    )
    return dists


# ==============================================================================
# PART 1 — STRUCTURE VALIDATION
# ==============================================================================

def validate_endpoints(name: str, init_atoms, final_atoms) -> dict:
    """
    Check that only the expected number of atoms have moved significantly
    between the initial (surface-bound) and final (dissolved) endpoints.

    Returns a dict with keys:
      - passed (bool)
      - expected_movers (int)
      - species (str)
      - moving_atoms (list of dicts: index, symbol, displacement)
      - stationary_warnings (list of dicts)
      - message (str)
    """
    result = {
        "name":                name,
        "passed":              False,
        "expected_movers":     None,
        "species":             None,
        "moving_atoms":        [],
        "stationary_warnings": [],
        "message":             "",
    }

    # 1. Atom count
    if len(init_atoms) != len(final_atoms):
        result["message"] = (
            f"Atom count mismatch: init={len(init_atoms)}, "
            f"final={len(final_atoms)}"
        )
        return result

    # 2. Element order
    init_syms  = init_atoms.get_chemical_symbols()
    final_syms = final_atoms.get_chemical_symbols()
    mismatches = [(i, s1, s2) for i, (s1, s2) in enumerate(zip(init_syms, final_syms)) if s1 != s2]
    if mismatches:
        result["message"] = (
            f"Element mismatch at {len(mismatches)} index(es): "
            + ", ".join(f"idx {i}: {s1}→{s2}" for i, s1, s2 in mismatches[:5])
        )
        return result

    # 3. Per-atom displacements
    dists = mic_displacements(init_atoms, final_atoms)

    movers = []
    stat_warnings = []
    for i, (sym, d) in enumerate(zip(init_syms, dists)):
        if d > DISSOLVING_THRESHOLD:
            movers.append({"index": i, "symbol": sym, "displacement_A": float(d)})
        elif d > STATIONARY_THRESHOLD:
            stat_warnings.append({"index": i, "symbol": sym, "displacement_A": float(d)})

    result["moving_atoms"]        = movers
    result["stationary_warnings"] = stat_warnings

    # 4. Compare against expected count
    species = parse_species_from_name(name)
    result["species"] = species

    if species is None:
        result["message"] = (
            f"Could not parse dissolving species from name '{name}'. "
            f"Known: {list(SPECIES_MOVERS.keys())}"
        )
        # Don't fail hard — just warn
        result["passed"] = True
        return result

    expected = SPECIES_MOVERS[species]
    result["expected_movers"] = expected
    n_moving = len(movers)

    if n_moving == expected:
        result["passed"] = True
        result["message"] = (
            f"OK — {n_moving}/{expected} atoms moved (species: {species})"
        )
    else:
        result["passed"] = False
        result["message"] = (
            f"MISMATCH — found {n_moving} moving atoms, expected {expected} "
            f"for species '{species}'"
        )

    return result


def print_validation_report(vr: dict):
    """Pretty-print a single validation result."""
    status = "[✓]" if vr["passed"] else "[✗]"
    print(f"\n  {status} {vr['name']}: {vr['message']}")

    if vr["moving_atoms"]:
        print(f"      Moving atoms (>{DISSOLVING_THRESHOLD} Å):")
        for a in vr["moving_atoms"]:
            print(f"        Index {a['index']:>4}  {a['symbol']}  "
                  f"Δr = {a['displacement_A']:.3f} Å")

    if vr["stationary_warnings"]:
        print(f"      Stationary warnings ({STATIONARY_THRESHOLD}–{DISSOLVING_THRESHOLD} Å):")
        for a in vr["stationary_warnings"][:5]:
            print(f"        Index {a['index']:>4}  {a['symbol']}  "
                  f"Δr = {a['displacement_A']:.3f} Å")
        if len(vr["stationary_warnings"]) > 5:
            print(f"        ... and {len(vr['stationary_warnings'])-5} more.")


def run_all_validations(configs: list, verbose=True) -> list:
    """Validate all configs and return list of validation result dicts."""
    print("\n" + "="*70)
    print("  PART 1 — ENDPOINT STRUCTURE VALIDATION")
    print("="*70)

    all_results = []
    for config in configs:
        name = config["name"]
        init_path  = config["initial"]
        final_path = config["final"]

        if not os.path.exists(init_path):
            print(f"  [✗] {name}: initial file not found: {init_path}")
            all_results.append({"name": name, "passed": False,
                                 "message": "File not found: " + init_path})
            continue
        if not os.path.exists(final_path):
            print(f"  [✗] {name}: final file not found: {final_path}")
            all_results.append({"name": name, "passed": False,
                                 "message": "File not found: " + final_path})
            continue

        init_atoms  = read(init_path)
        final_atoms = read(final_path)
        vr = validate_endpoints(name, init_atoms, final_atoms)

        if verbose:
            print_validation_report(vr)

        all_results.append(vr)

    n_pass = sum(1 for r in all_results if r["passed"])
    n_fail = len(all_results) - n_pass
    print(f"\n  Validation summary: {n_pass} passed, {n_fail} failed\n")
    return all_results


# ==============================================================================
# PART 2 — MULTI-MODEL NEB
# ==============================================================================

def load_models() -> dict:
    """Load all MACE calculators and return {model_key: calculator}."""
    calcs = {}
    for key, cfg in MODELS.items():
        print(f"[→] Loading model '{cfg['label']}' from: {cfg['path']}")
        calcs[key] = MACECalculator(
            model_paths=cfg["path"],
            device=DEVICE,
            default_dtype=DTYPE,
        )
        print(f"[✓] Loaded: {cfg['label']}")
    return calcs


def run_neb_single_model(
    init_atoms,
    final_atoms,
    calc,
    name: str,
    model_key: str,
    output_dir: str,
) -> dict:
    """
    Run NEB for a single model on a single config.

    Returns a result dict with energies, barriers, convergence info,
    and the list of image Atoms objects.
    """
    model_label = MODELS[model_key]["label"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  [→] NEB: {name}  |  model: {model_label}")

    result = {
        "name":        name,
        "model_key":   model_key,
        "model_label": model_label,
        "converged":   False,
        "images":      [],
        "energies":    [],
        "max_forces":  [],
        "barrier_forward":  None,
        "barrier_reverse":  None,
        "reaction_energy":  None,
        "ts_image_index":   None,
        "E_initial":        None,
        "E_final":          None,
        "E_ts":             None,
        "elapsed_s":        None,
        "error":            None,
    }

    # ── Build image list ──────────────────────────────────────────────────────
    images = [init_atoms.copy()]
    for _ in range(N_IMAGES):
        images.append(init_atoms.copy())
    images.append(final_atoms.copy())

    #neb = NEB(images, climb=CLIMB, allow_shared_calculator=True)
    neb = NEB(images, climb=CLIMB, allow_shared_calculator=True)
    neb.interpolate(apply_constraint=False)

    fixed_indices = get_fixed_indices(init_atoms)
    for image in images:
        if fixed_indices:
            image.set_constraint(FixAtoms(indices=fixed_indices))
        image.calc = calc

    # ── Optimise ──────────────────────────────────────────────────────────────
    traj_path = os.path.join(output_dir, f"{name}_neb.traj")
    log_path  = os.path.join(output_dir, f"{name}_neb.log")

    if NEB_OPTIMIZER == "FIRE":
        optimizer = FIRE(neb, trajectory=traj_path, logfile=log_path)
    else:
        optimizer = BFGS(neb, trajectory=traj_path, logfile=log_path)

    t0 = time.perf_counter()
    try:
        converged = optimizer.run(fmax=NEB_FMAX, steps=NEB_MAX_STEPS)
        result["converged"] = converged
        print(f"    [{'✓' if converged else '~'}] NEB {'converged' if converged else 'did not converge'} "
              f"in {time.perf_counter()-t0:.1f} s")
    except Exception as e:
        result["error"] = str(e)
        print(f"    [✗] NEB failed: {e}")
        return result

    result["elapsed_s"] = time.perf_counter() - t0

    # ── Extract energies and forces ───────────────────────────────────────────
    energies   = []
    max_forces = []
    for image in images:
        try:
            e = image.get_potential_energy()
            f = np.sqrt((image.get_forces() ** 2).sum(axis=1)).max()
        except Exception:
            e, f = np.nan, np.nan
        energies.append(e)
        max_forces.append(f)

    result["images"]     = images
    result["energies"]   = energies
    result["max_forces"] = max_forces

    # ── Barrier analysis ──────────────────────────────────────────────────────
    E_initial = energies[0]
    E_final   = energies[-1]
    intermediate = [(i, e) for i, e in enumerate(energies[1:-1], start=1)
                    if not np.isnan(e)]

    if intermediate:
        ts_idx, E_ts = max(intermediate, key=lambda x: x[1])
        result["E_initial"]       = E_initial
        result["E_final"]         = E_final
        result["E_ts"]            = E_ts
        result["ts_image_index"]  = ts_idx
        result["barrier_forward"] = E_ts - E_initial
        result["barrier_reverse"] = E_ts - E_final
        result["reaction_energy"] = E_final - E_initial

        print(f"    E_barrier (fwd) = {result['barrier_forward']:+.4f} eV  "
              f"|  ΔE = {result['reaction_energy']:+.4f} eV  "
              f"|  TS @ image {ts_idx}")

    # ── Save extxyz ───────────────────────────────────────────────────────────
    for idx, img in enumerate(images):
        img.info.pop("energy",      None)
        img.info.pop("free_energy", None)
        img.arrays.pop("forces",    None)
        img.arrays.pop("energies",  None)
        img.info["system_type"]  = name
        img.info["neb_image"]    = idx
        img.info["model"]        = model_label
        img.info["source"]       = "mace_neb_compare"

    xyz_path = os.path.join(output_dir, f"{name}_neb.extxyz")
    write(xyz_path, images, format="extxyz")

    # ── Plot individual energy profile ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        nebtools = NEBTools(images)
        nebtools.plot_band(ax=ax)
    except Exception:
        E_rel = np.array(energies) - energies[0]
        ax.plot(range(len(E_rel)), E_rel, "o-")
        ax.set_ylabel("ΔE (eV)")
    ax.set_title(f"{name} — {model_label}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{name}_neb_profile.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return result

def run_all_nebs(configs: list, calcs: dict, validation_results: list, no_neb: bool = False) -> dict:
    """
    Run NEB for every config × every model.
    If no_neb=True, only relaxes endpoints (if RELAX_ENDPOINTS is set) and skips NEB.
    Returns nested dict: neb_results[config_name][model_key] = result_dict
    """
    print("\n" + "="*70)
    print("  PART 2 — MULTI-MODEL NEB SIMULATIONS")
    print("="*70)

    # Build a quick lookup for validation pass/fail
    val_lookup = {r["name"]: r for r in validation_results}

    neb_results = {}
    for config in configs:
        name       = config["name"]
        init_path  = config["initial"]
        final_path = config["final"]

        vr = val_lookup.get(name, {})
        if ABORT_ON_VALIDATION_FAIL and not vr.get("passed", True):
            print(f"\n  [!] Skipping NEB for {name}: failed validation.")
            continue

        if not (os.path.exists(init_path) and os.path.exists(final_path)):
            print(f"\n  [!] Skipping NEB for {name}: missing structure file(s).")
            continue

        init_atoms  = read(init_path)
        final_atoms = read(final_path)
        
        if RELAX_ENDPOINTS:
            print(f"\n      [→] Pre-relaxing endpoints for {name} via MACE-V4...")
            target_calc = calcs["finetuned"]  # Maps to your MACE-V4 model
            
            for label, atoms in [("Initial", init_atoms), ("Final", final_atoms)]:
                atoms.calc = target_calc
                traj_path  = os.path.join(OUTPUT_ROOT, f"{name}_{label}_relax.traj")
                # Using FIRE or BFGS; logfile=None keeps your terminal clean
                opt = FIRE(atoms, logfile=None, trajectory=traj_path)
                t_opt0 = time.perf_counter()
                opt.run(fmax=0.05, steps=200) 
                
                # Check maximum remaining force
                f_max = np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max()
                print(f"      [✓] {label} optimized ({time.perf_counter()-t_opt0:.1f}s). Max force: {f_max:.4f} eV/Å")
            
            out_ep = os.path.join(OUTPUT_ROOT, "relaxed_endpoints")
            os.makedirs(out_ep, exist_ok=True)
            write(os.path.join(out_ep, f"{name}_initial_relaxed.xyz"), init_atoms)
            write(os.path.join(out_ep, f"{name}_final_relaxed.xyz"),   final_atoms)
            print(f"      [✓] Saved relaxed endpoints to: {out_ep}/")
            # ──────────────────────────────────────────────────────────────────
            
            print(f"      [✓] Endpoints stabilized. Proceeding to multi-model NEB loops.")

        if no_neb:
            print(f"      [!] --no-neb set: skipping NEB for {name}.")
            neb_results[name] = {}
            continue

        print(f"      [✓] Endpoints stabilized. Proceeding to multi-model NEB.")
        neb_results[name] = {}
        for model_key, calc in calcs.items():
            out_dir = os.path.join(OUTPUT_ROOT, model_key, name)
            result  = run_neb_single_model(
                init_atoms.copy(), final_atoms.copy(),
                calc, name, model_key, out_dir,
            )
            neb_results[name][model_key] = result

    return neb_results

# ==============================================================================
# PART 3 — MODEL COMPARISON
# ==============================================================================

def compare_models(neb_results: dict) -> list:
    """
    For each config where both models ran successfully, compute comparison
    metrics and return a list of comparison dicts.

    Metrics:
      barrier_forward_diff    (eV)        — ΔE‡_fwd(FT) − ΔE‡_fwd(Found.)
      barrier_reverse_diff    (eV)
      reaction_energy_diff    (eV)
      ts_image_shift          (images)    — where each model puts the TS
      energy_rmse             (eV)        — per-image energy RMSE between models
      energy_mae              (eV)        — per-image energy MAE
      max_force_rmse          (eV/Å)     — per-image max-force RMSE
      barrier_pct_diff        (%)         — % change in forward barrier
    """
    print("\n" + "="*70)
    print("  PART 3 — MODEL COMPARISON")
    print("="*70)

    keys = list(MODELS.keys())
    if len(keys) < 2:
        print("  [!] Need at least 2 models to compare.")
        return []

    k1, k2 = keys[0], keys[1]   # foundational, finetuned
    comparisons = []

    for name, model_results in neb_results.items():
        if k1 not in model_results or k2 not in model_results:
            continue
        r1 = model_results[k1]
        r2 = model_results[k2]

        # Skip if either failed
        if r1.get("error") or r2.get("error"):
            print(f"  [!] {name}: one or both models errored — skipping comparison.")
            continue
        if not r1["energies"] or not r2["energies"]:
            continue

        E1 = np.array(r1["energies"])
        E2 = np.array(r2["energies"])
        F1 = np.array(r1["max_forces"])
        F2 = np.array(r2["max_forces"])

        # Align reference to initial image energy (relative energies)
        E1_rel = E1 - E1[0]
        E2_rel = E2 - E2[0]

        n = min(len(E1), len(E2))
        energy_rmse    = float(np.sqrt(np.mean((E1_rel[:n] - E2_rel[:n])**2)))
        energy_mae     = float(np.mean(np.abs(E1_rel[:n] - E2_rel[:n])))
        force_rmse     = float(np.sqrt(np.nanmean((F1[:n] - F2[:n])**2)))

        bf1 = r1["barrier_forward"]
        bf2 = r2["barrier_forward"]
        br1 = r1["barrier_reverse"]
        br2 = r2["barrier_reverse"]
        re1 = r1["reaction_energy"]
        re2 = r2["reaction_energy"]
        ts1 = r1["ts_image_index"]
        ts2 = r2["ts_image_index"]

        barrier_pct = None
        if bf1 is not None and bf2 is not None and bf1 != 0:
            barrier_pct = 100.0 * (bf2 - bf1) / abs(bf1)

        comp = {
            "name":                    name,
            "model_1":                 MODELS[k1]["label"],
            "model_2":                 MODELS[k2]["label"],
            "barrier_forward_m1":      bf1,
            "barrier_forward_m2":      bf2,
            "barrier_forward_diff":    (bf2 - bf1) if (bf1 is not None and bf2 is not None) else None,
            "barrier_reverse_m1":      br1,
            "barrier_reverse_m2":      br2,
            "barrier_reverse_diff":    (br2 - br1) if (br1 is not None and br2 is not None) else None,
            "reaction_energy_m1":      re1,
            "reaction_energy_m2":      re2,
            "reaction_energy_diff":    (re2 - re1) if (re1 is not None and re2 is not None) else None,
            "ts_image_m1":             ts1,
            "ts_image_m2":             ts2,
            "ts_image_shift":          (ts2 - ts1) if (ts1 is not None and ts2 is not None) else None,
            "energy_rmse_eV":          energy_rmse,
            "energy_mae_eV":           energy_mae,
            "max_force_rmse_eVA":      force_rmse,
            "barrier_pct_diff":        barrier_pct,
            "energies_m1":             E1_rel.tolist(),
            "energies_m2":             E2_rel.tolist(),
            "max_forces_m1":           F1.tolist(),
            "max_forces_m2":           F2.tolist(),
        }
        comparisons.append(comp)

        # ── Print comparison table ────────────────────────────────────────────
        l1 = MODELS[k1]["label"]
        l2 = MODELS[k2]["label"]
        print(f"\n  {'─'*60}")
        print(f"  {name}")
        print(f"  {'─'*60}")
        print(f"  {'Metric':<35} {l1:>12}  {l2:>12}  {'Δ':>10}")
        print(f"  {'─'*60}")

        def row(label, v1, v2, diff, fmt=".4f", unit=""):
            v1s  = f"{v1:{fmt}}{unit}" if v1  is not None else "N/A"
            v2s  = f"{v2:{fmt}}{unit}" if v2  is not None else "N/A"
            ds   = f"{diff:+{fmt}}{unit}" if diff is not None else "—"
            print(f"  {label:<35} {v1s:>12}  {v2s:>12}  {ds:>10}")

        row("Forward barrier (eV)",     bf1, bf2,
            comp["barrier_forward_diff"])
        row("Reverse barrier (eV)",     br1, br2,
            comp["barrier_reverse_diff"])
        row("Reaction energy ΔE (eV)",  re1, re2,
            comp["reaction_energy_diff"])
        row("TS image index",           ts1, ts2,
            comp["ts_image_shift"], fmt=".0f")
        print(f"  {'Energy RMSE (rel, eV)':<35} {'':>12}  {'':>12}  {energy_rmse:>10.4f}")
        print(f"  {'Energy MAE  (rel, eV)':<35} {'':>12}  {'':>12}  {energy_mae:>10.4f}")
        print(f"  {'Max-force RMSE (eV/Å)':<35} {'':>12}  {'':>12}  {force_rmse:>10.4f}")
        if barrier_pct is not None:
            print(f"  {'Barrier % diff (fwd)':<35} {'':>12}  {'':>12}  {barrier_pct:>+10.1f}%")

    return comparisons


def plot_comparison(comparisons: list, neb_results: dict):
    """
    For each config: overlay energy profiles of both models on one axis.
    Then produce a summary bar chart of forward barriers across all configs.
    """
    cmp_dir = os.path.join(OUTPUT_ROOT, "comparison_plots")
    os.makedirs(cmp_dir, exist_ok=True)

    keys   = list(MODELS.keys())
    colors = [MODELS[k]["color"] for k in keys]
    labels = [MODELS[k]["label"] for k in keys]

    # ── Per-config overlay plots ──────────────────────────────────────────────
    for comp in comparisons:
        name = comp["name"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Model comparison — {name}", fontsize=13, fontweight="bold")

        # Energy profiles
        ax = axes[0]
        for k, c, lbl in zip(keys, colors, labels):
            r = neb_results[name].get(k, {})
            E = np.array(r.get("energies", []))
            if len(E):
                ax.plot(range(len(E)), E - E[0], "o-", color=c, label=lbl,
                        linewidth=2, markersize=5)
        ax.set_xlabel("NEB image")
        ax.set_ylabel("Relative energy (eV)")
        ax.set_title("Energy profile")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        # Max force per image
        ax = axes[1]
        for k, c, lbl in zip(keys, colors, labels):
            r = neb_results[name].get(k, {})
            F = r.get("max_forces", [])
            if F:
                ax.plot(range(len(F)), F, "s--", color=c, label=lbl,
                        linewidth=1.5, markersize=4, alpha=0.8)
        ax.axhline(NEB_FMAX, color="gray", linestyle=":", linewidth=1,
                   label=f"NEB F_max threshold ({NEB_FMAX} eV/Å)")
        ax.set_xlabel("NEB image")
        ax.set_ylabel("Max force (eV/Å)")
        ax.set_title("Max force per image")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(cmp_dir, f"{name}_comparison.png"), dpi=150)
        plt.close()

    # ── Summary bar chart — forward barriers ─────────────────────────────────
    if comparisons:
        names  = [c["name"] for c in comparisons]
        x      = np.arange(len(names))
        width  = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(names)*1.5), 5))

        for j, (k, c, lbl) in enumerate(zip(keys, colors, labels)):
            barriers = [c2.get(f"barrier_forward_m{j+1}") for c2 in comparisons]
            barriers_plot = [b if b is not None else 0.0 for b in barriers]
            ax.bar(x + j*width, barriers_plot, width, label=lbl,
                   color=c, alpha=0.8, edgecolor="white")

        ax.set_xticks(x + width/2)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Forward barrier (eV)")
        ax.set_title("Forward dissolution barriers — model comparison")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(cmp_dir, "summary_barriers.png"), dpi=150)
        plt.close()
        print(f"\n[✓] Summary barrier plot saved.")

    print(f"[✓] Comparison plots saved to: {cmp_dir}/")


# ==============================================================================
# PART 4 — NEB PATHOLOGY DETECTION
# ==============================================================================

def detect_pathologies(neb_results: dict) -> dict:
    """
    Scan every NEB trajectory (per model, per config) for signs that the
    MACE model is performing badly.

    Flags per image:
      - ENERGY_SPIKE : energy jump > PATHOLOGY_ENERGY_SPIKE eV vs prior image
      - ENERGY_ABS   : absolute energy > E_initial + PATHOLOGY_ENERGY_ABS eV
      - FORCE_HIGH   : max force > PATHOLOGY_FORCE_FMAX eV/Å
      - ATOM_DISP    : any atom displaced > PATHOLOGY_ATOM_DISP Å from image 0

    Returns nested dict: pathologies[config_name][model_key] = list of flags
    """
    print("\n" + "="*70)
    print("  PART 4 — NEB PATHOLOGY DETECTION")
    print("="*70)

    all_pathologies = {}
    any_found = False

    for name, model_results in neb_results.items():
        all_pathologies[name] = {}

        for model_key, result in model_results.items():
            model_label = MODELS[model_key]["label"]
            images      = result.get("images", [])
            energies    = result.get("energies", [])
            max_forces  = result.get("max_forces", [])

            if not images or not energies:
                all_pathologies[name][model_key] = []
                continue

            flags = []
            E0    = energies[0]
            init_positions = images[0].positions.copy()
            cell  = images[0].get_cell()
            pbc   = images[0].get_pbc()

            for i, (image, E, Fmax) in enumerate(zip(images, energies, max_forces)):

                image_flags = []

                # ENERGY_SPIKE — sudden jump between consecutive images
                if i > 0:
                    prev_E = energies[i - 1]
                    if not (np.isnan(E) or np.isnan(prev_E)):
                        if abs(E - prev_E) > PATHOLOGY_ENERGY_SPIKE:
                            image_flags.append({
                                "type": "ENERGY_SPIKE",
                                "value": float(E - prev_E),
                                "threshold": PATHOLOGY_ENERGY_SPIKE,
                                "detail": f"ΔE = {E-prev_E:+.3f} eV vs image {i-1}",
                            })

                # ENERGY_ABS — implausibly high absolute energy
                if not np.isnan(E) and (E - E0) > PATHOLOGY_ENERGY_ABS:
                    image_flags.append({
                        "type": "ENERGY_ABS",
                        "value": float(E - E0),
                        "threshold": PATHOLOGY_ENERGY_ABS,
                        "detail": f"E − E_init = {E-E0:+.3f} eV",
                    })

                # FORCE_HIGH — forces far above NEB convergence
                if not np.isnan(Fmax) and Fmax > PATHOLOGY_FORCE_FMAX:
                    image_flags.append({
                        "type": "FORCE_HIGH",
                        "value": float(Fmax),
                        "threshold": PATHOLOGY_FORCE_FMAX,
                        "detail": f"F_max = {Fmax:.3f} eV/Å",
                    })

                # ATOM_DISP — any atom moved far from its image-0 position
                if i > 0:
                    _, dists = find_mic(
                        image.positions - init_positions,
                        cell, pbc,
                    )
                    max_disp_atom = int(np.argmax(dists))
                    max_disp      = float(dists[max_disp_atom])
                    sym           = image.get_chemical_symbols()[max_disp_atom]
                    if max_disp > PATHOLOGY_ATOM_DISP:
                        image_flags.append({
                            "type":      "ATOM_DISP",
                            "value":     max_disp,
                            "threshold": PATHOLOGY_ATOM_DISP,
                            "detail":    f"Atom {max_disp_atom} ({sym}) Δr = {max_disp:.2f} Å from image 0",
                        })

                if image_flags:
                    flags.append({
                        "structure": name,
                        "model":     model_label,
                        "image":     i,
                        "flags":     image_flags,
                    })
                    any_found = True

            all_pathologies[name][model_key] = flags

    # ── Pretty-print all flagged images ───────────────────────────────────────
    if not any_found:
        print("\n  [✓] No pathologies detected across all NEBs.\n")
    else:
        print("\n  [!] Pathologies detected:\n")
        for name, by_model in all_pathologies.items():
            for model_key, flags in by_model.items():
                if not flags:
                    continue
                model_label = MODELS[model_key]["label"]
                for f in flags:
                    for flag in f["flags"]:
                        print(f"  ⚠  {name}  |  {model_label}  |  image {f['image']:>2}  "
                              f"|  {flag['type']:<15}  {flag['detail']}")
        print()

    return all_pathologies


def plot_pathology_summary(all_pathologies: dict, neb_results: dict):
    """
    For each config × model, plot energy profile with pathological images
    highlighted so problems are immediately visible.
    """
    path_dir = os.path.join(OUTPUT_ROOT, "pathology_plots")
    os.makedirs(path_dir, exist_ok=True)

    for name, by_model in all_pathologies.items():
        has_anything = any(flags for flags in by_model.values())
        if not has_anything:
            continue

        fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 5),
                                  sharey=False)
        if len(MODELS) == 1:
            axes = [axes]

        fig.suptitle(f"Pathology map — {name}", fontsize=13, fontweight="bold")

        for ax, (model_key, flags) in zip(axes, by_model.items()):
            model_label = MODELS[model_key]["label"]
            color       = MODELS[model_key]["color"]
            r           = neb_results[name].get(model_key, {})
            energies    = r.get("energies", [])

            if not energies:
                ax.set_title(f"{model_label} (no data)")
                continue

            E_rel = np.array(energies) - energies[0]
            ax.plot(range(len(E_rel)), E_rel, "o-", color=color,
                    linewidth=2, markersize=5, label="Energy profile")

            # Collect flagged image indices and their worst flag type
            flag_priority = {"ENERGY_ABS": 4, "ENERGY_SPIKE": 3,
                             "FORCE_HIGH": 2, "ATOM_DISP": 1}
            flagged_images = {}
            for f in flags:
                idx = f["image"]
                for flag in f["flags"]:
                    current_priority = flag_priority.get(flag["type"], 0)
                    if idx not in flagged_images or current_priority > flagged_images[idx][1]:
                        flagged_images[idx] = (flag["type"], current_priority, flag["detail"])

            flag_colors = {
                "ENERGY_ABS":   "#FF0000",
                "ENERGY_SPIKE": "#FF6600",
                "FORCE_HIGH":   "#9C27B0",
                "ATOM_DISP":    "#FF9800",
            }
            for idx, (ftype, _, detail) in flagged_images.items():
                if idx < len(E_rel):
                    fc = flag_colors.get(ftype, "red")
                    ax.scatter(idx, E_rel[idx], s=120, color=fc, zorder=5,
                               label=f"Image {idx}: {ftype}")
                    ax.annotate(f"img {idx}\n{ftype}",
                                xy=(idx, E_rel[idx]),
                                xytext=(idx + 0.3, E_rel[idx] + 0.05),
                                fontsize=7, color=fc)

            ax.set_xlabel("NEB image")
            ax.set_ylabel("ΔE (eV)")
            ax.set_title(f"{model_label}")
            ax.grid(True, linestyle="--", alpha=0.4)

            # Deduplicate legend
            handles, lbls = ax.get_legend_handles_labels()
            by_lbl = dict(zip(lbls, handles))
            ax.legend(by_lbl.values(), by_lbl.keys(), fontsize=7)

        plt.tight_layout()
        out_path = os.path.join(path_dir, f"{name}_pathology.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[✓] Pathology plot: {out_path}")
# ==============================================================================
# PART 5 — ACTIVE DFT NEB PATHWAY REFINEMENT (LOCAL POLISH)
# ==============================================================================

RUN_DFT_REFINEMENT = True  # Toggle off if you want to skip DFT completely
DFT_COMMAND        = "mpiexec -n 6 vasp_std"  # Command template for running DFT NEB refinement (expects {input_dir} and {output_dir} placeholders)

# Optimized for a quick, localized path relaxation
DFT_PARAMS = {
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

def run_dft_path_refinement(name: str, mace_images: list, target_model_label: str) -> list:
    """
    Takes the finalized MACE NEB pathway, binds unique VASP calculators to the 
    interstitial images in isolated folders, and runs a localized DFT-NEB relaxation.
    """
    from ase.calculators.vasp import Vasp
    from ase.optimize import MDMin
    
    print(f"\n  [→] DFT Refinement: Polishing pathway from '{target_model_label}' via VASP NEB...")
    
    # Deep copy the MACE path to prevent altering the baseline MACE structures in memory
    refined_images = [img.copy() for img in mace_images]
    cfg_dft_root   = os.path.join(OUTPUT_ROOT, "dft_refinement", name)
    
    # Assign unique working folders to each moving image to avoid parallel file collisions
    for idx, atoms in enumerate(refined_images[1:-1], start=1):
        image_dir = os.path.join(cfg_dft_root, f"image_{idx:02d}")
        os.makedirs(image_dir, exist_ok=True)
        
        calc = Vasp(
            command=DFT_COMMAND,
            directory=image_dir,
            **DFT_PARAMS
        )
        atoms.calc = calc

    # Re-instantiate the NEB string context for ASE using the DFT-linked images
    # We match your script's settings (climb=False, k=0.1)
    dft_neb = NEB(refined_images, climb=False, k=0.1)
    
    traj_path = os.path.join(cfg_dft_root, f"{name}_dft_refined.traj")
    log_path  = os.path.join(cfg_dft_root, f"{name}_dft_refined.log")
    
    # Using MDMin to coordinate the overarching NEB image-string translations
    optimizer = MDMin(dft_neb, trajectory=traj_path, logfile=log_path)
    
    t0 = time.perf_counter()
    try:
        # We run the optimizer loop. It stops when the image forces drop below 
        # 0.05 eV/Å or when VASP hitting its internal NSW step cap cuts the loop.
        print(f"      Running VASP-driven string optimization (Max VASP NSW={DFT_PARAMS['nsw']})...")
        optimizer.run(fmax=0.05, steps=DFT_PARAMS['nsw'])
        print(f"    [✓] DFT path refinement finished in {time.perf_counter()-t0:.1f} s")
    except Exception as err:
        print(f"    [✗] VASP NEB refinement errored out: {err}")
        return []
        
    return refined_images


def plot_refined_dft_overlay(name: str, comp_dict: dict, refined_images: list):
    """Plots the newly relaxed DFT pathway alongside the original MACE models."""
    if not refined_images:
        return
        
    plot_dir = os.path.join(OUTPUT_ROOT, "comparison_plots")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # MACE Baseline Curves
    ax.plot(range(len(comp_dict["energies_m1"])), comp_dict["energies_m1"], "o-", 
            color=MODELS["foundational"]["color"], label=comp_dict["model_1"], alpha=0.6)
    ax.plot(range(len(comp_dict["energies_m2"])), comp_dict["energies_m2"], "o-", 
            color=MODELS["finetuned"]["color"], label=comp_dict["model_2"], alpha=0.6)
    
    # Read true calculated energies out of the relaxed DFT images
    try:
        dft_energies = [img.get_potential_energy() for img in refined_images]
        dft_rel = np.array(dft_energies) - dft_energies[0]
        
        ax.plot(range(len(dft_rel)), dft_rel, "D--", color="#111111", 
                label="VASP Refined Path", linewidth=2.5, markersize=6)
    except Exception:
        pass
        
    ax.set_xlabel("NEB Image Index")
    ax.set_ylabel("Relative Energy (eV)")
    ax.set_title(f"Pathway Polishing: MACE Baseline vs. VASP Refined\nSystem: {name}", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_dft_refined_overlay.png"), dpi=150)
    plt.close()


# ==============================================================================
# OUTPUT & REPORTING
# ==============================================================================

def save_json_report(
    validation_results: list,
    comparisons: list,
    all_pathologies: dict,
):
    """Save all results to a machine-readable JSON report."""
    report_dir = os.path.join(OUTPUT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Pathologies are nested dicts of lists — already JSON-serialisable
    report = {
        "run_timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "models":           {k: {"label": v["label"], "path": v["path"]}
                             for k, v in MODELS.items()},
        "validation":       validation_results,
        "comparisons":      [
            {k: v for k, v in c.items()
             if k not in ("energies_m1", "energies_m2",
                          "max_forces_m1", "max_forces_m2")}
            for c in comparisons
        ],
        "pathologies":      {
            name: {
                mk: flags for mk, flags in by_model.items()
            }
            for name, by_model in all_pathologies.items()
        },
    }

    path = os.path.join(report_dir, "neb_comparison_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[✓] JSON report saved: {path}")


def print_final_summary(comparisons: list, all_pathologies: dict):
    """Print a compact final summary table."""
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    keys = list(MODELS.keys())
    if len(keys) >= 2:
        k1, k2 = keys[0], keys[1]
        l1, l2 = MODELS[k1]["label"], MODELS[k2]["label"]

        print(f"\n  {'Config':<22} {'ΔE‡_fwd '+l1:>14}  {'ΔE‡_fwd '+l2:>14}  "
              f"{'Δ(eV)':>8}  {'RMSE':>8}  {'Pathol.'}")
        print("  " + "─"*80)

        for comp in comparisons:
            name  = comp["name"]
            bf1   = comp["barrier_forward_m1"]
            bf2   = comp["barrier_forward_m2"]
            diff  = comp["barrier_forward_diff"]
            rmse  = comp["energy_rmse_eV"]

            # Count total pathology flags across models
            n_path = sum(
                len(flags)
                for by_model in [all_pathologies.get(name, {})]
                for flags in by_model.values()
            )

            bf1s  = f"{bf1:+.3f}" if bf1 is not None else "N/A"
            bf2s  = f"{bf2:+.3f}" if bf2 is not None else "N/A"
            diffs = f"{diff:+.3f}" if diff is not None else "N/A"
            rmses = f"{rmse:.4f}"  if rmse is not None else "N/A"
            paths = f"{n_path} flags" if n_path else "clean"

            print(f"  {name:<22} {bf1s:>14}  {bf2s:>14}  {diffs:>8}  {rmses:>8}  {paths}")

    print("\n" + "="*70 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global OUTPUT_ROOT

    parser = argparse.ArgumentParser(
        description="MACE multi-model NEB comparison for Pt dissolution pathways"
    )
    parser.add_argument("--csv",           type=str, default=PATH_CSV,
                        help="Path to config CSV (Name, initial, final)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run structure validation, skip NEB")
    parser.add_argument("--no-neb",        action="store_true",
                        help="Skip NEB runs (useful to re-run analysis on existing results)")
    parser.add_argument("--output-dir",    type=str, default=OUTPUT_ROOT,
                        help="Root output directory")
    args = parser.parse_args()

    OUTPUT_ROOT = args.output_dir
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("\n" + "="*70)
    print("  MACE MULTI-MODEL NEB COMPARISON")
    print("="*70)
    print(f"  Models:     {', '.join(m['label'] for m in MODELS.values())}")
    print(f"  CSV:        {args.csv}")
    print(f"  Output dir: {OUTPUT_ROOT}")
    print(f"  Started:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Load configs ──────────────────────────────────────────────────────────
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        configs = [
            {"name": row["Name"], "initial": row["initial"], "final": row["final"]}
            for row in reader
        ]
    print(f"[✓] Loaded {len(configs)} configs from CSV.\n")

    # ── Part 1: Validation ────────────────────────────────────────────────────
    validation_results = run_all_validations(configs)

    if args.validate_only:
        save_json_report(validation_results, [], {})
        print("[✓] Validation-only run complete.")
        return

    # ── Load models ───────────────────────────────────────────────────────────
    calcs = load_models()

    # ── Part 2: NEB simulations ───────────────────────────────────────────────
    neb_results = {}
    if not args.no_neb:
        neb_results = run_all_nebs(configs, calcs, validation_results, no_neb=args.no_neb)
    else:
        print("\n[!] --no-neb set: skipping NEB runs.\n")

    if not neb_results:
        print("[!] No NEB results to analyse. Exiting.")
        save_json_report(validation_results, [], {})
        return

    # ── Part 3: Comparison ────────────────────────────────────────────────────
    comparisons = compare_models(neb_results)
    plot_comparison(comparisons, neb_results)

    # ── Part 4: Pathology detection ───────────────────────────────────────────
    all_pathologies = detect_pathologies(neb_results)
    plot_pathology_summary(all_pathologies, neb_results)

    # ── Part 5: DFT refinement (optional) ─────────────────────────────────────
    if RUN_DFT_REFINEMENT:
        try:
            print("\n" + "="*70)
            print("  PART 5 — ACTIVE DFT PATHWAY REFINEMENT")
            print("="*70)
            
            for comp in comparisons:
                name = comp["name"]
                
                # Select the images from your fine-tuned model to serve as the initial guess path
                if name in neb_results and "finetuned" in neb_results[name]:
                    mace_images = neb_results[name]["finetuned"].get("images")
                    model_lbl   = neb_results[name]["finetuned"].get("model_label")
                    
                    if mace_images:
                        # Run the VASP NEB local relaxation sweep
                        refined_dft_images = run_dft_path_refinement(name, mace_images, model_lbl)
                        
                        if refined_dft_images:
                            # Extract and print final DFT energies and barrier definitions
                            try:
                                dft_tools = NEBTools(refined_dft_images)
                                fwd_b, rev_b = dft_tools.get_barrier(fit=False)
                                print(f"\n  ✨  Final DFT-Refined Barrier Results for {name}:")
                                print(f"      • True DFT Forward Barrier: {fwd_b:.4f} eV")
                                print(f"      • True DFT Reverse Barrier: {rev_b:.4f} eV")
                                print(f"      • Initial MACE Guess Error: {comp['barrier_forward_m2'] - fwd_b:+.4f} eV")
                            except Exception as e:
                                print(f"      [~] Could not parse exact barriers with NEBTools: {e}")
                            
                            # Generate the comparison diagram overlaying the true local minimum points
                            plot_refined_dft_overlay(name, comp, refined_dft_images)
        except Exception as e:
            print(f"\n[✗] DFT refinement process encountered an error: {e}")
            print("    Skipping DFT refinement and proceeding to final reporting.\n")
            

    # ── Save reports ──────────────────────────────────────────────────────────    save_json_report(validation_results, comparisons, all_pathologies)
    print_final_summary(comparisons, all_pathologies)

    print(f"[✓] All done. Results in: {OUTPUT_ROOT}/")
    print(f"    ├── foundational/       ← NEB files for model 1")
    print(f"    ├── finetuned/          ← NEB files for model 2")
    print(f"    ├── comparison_plots/   ← overlay energy profiles + barrier bar charts")
    print(f"    ├── pathology_plots/    ← flagged image plots")
    print(f"    └── reports/            ← neb_comparison_report.json")


if __name__ == "__main__":
    main()
