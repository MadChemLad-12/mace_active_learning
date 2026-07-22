#!/usr/bin/env python3
"""
pipeline_audit.py
=================
Tracks raw structures or multi-frame NEB pathways through the entire MACE AL 
pipeline to identify precisely where structures fell off or were skipped.

Features:
  - Supports .cif, .xyz, and multi-frame NEB trajectory (.extxyz) files.
  - Applies Farthest Point Sampling (FPS) on multi-frame pathways.
  - Provides `--force-cp2k` to automatically construct missing CP2K inputs.
  - NEW: Provides `--add-master` and `--add-clean` to manually force-parse 
    completed CP2K .out files directly into your primary datasets.

Usage:
  # Standard Tracking Audit
  python pipeline_audit.py --audit geo_opt_results/Chair4x4sat_sol_initial_opt.cif
  
  # Force-Parse finished CP2K outputs directly into the master pool
  python pipeline_audit.py --add-master cp2k_sp_round4/sp_Dry0.375PtOH_r4_0004.out
  
  # Force-Parse finished CP2K outputs directly into the clean training pool
  python pipeline_audit.py --add-clean cp2k_sp_round4/sp_Dry0.375PtOH_r4_0015.out
"""

import os
import re
import argparse
import hashlib
import numpy as np
from pathlib import Path
import ase.io
from ase.io import read, write
from ase.units import Hartree, Bohr, eV, Angstrom
from sklearn.preprocessing import normalize
from ase.config import cfg
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from patches import apply_dftd3_cell_patch
apply_dftd3_cell_patch()

# Import functions and configurations from your active learning pipeline ecosystem
from MACE_CP2K_pipeline.src.active_pipeline  import (
    APPLY_D3, write_cp2k_sp, POOL_FILE, CP2K_TIMEOUT,
    parse_cell_from_out, parse_positions_from_out, 
    _write_submission_script, parse_stress_from_out
)
ROUND = 1
HASH_PRECISION = 4
def get_atoms_hash(atoms):
    """Generates a stable geometric MD5 hash matching your pipeline profile."""
    pos_data = np.round(atoms.get_positions(), HASH_PRECISION).tobytes()
    nuc_data = atoms.get_atomic_numbers().tobytes()
    return hashlib.md5(pos_data + nuc_data).hexdigest()

def parse_energy_forces(out_content):
    """
    Parses total energy and atomic forces from CP2K .out content.
    Returns:
        energy (float or None): Energy in eV
        forces (np.ndarray or None): Nx3 array of forces in eV/A
    """
    energy = None
    forces = None
    
    energy_match = re.search(r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-\d.]+)", out_content)
    if energy_match:
        energy = float(energy_match.group(1)) * Hartree
    else:
        print("  [!] Warning: Total energy not found in CP2K output.")
    
    forces_match = re.search(r"ATOMIC FORCES in \[a\.u\.\](.*?)SUM OF ATOMIC FORCES", out_content, re.DOTALL)
    if forces_match:
        try:
            forces_str = forces_match.group(1).strip()
            forces_lines = forces_str.splitlines()
            forces = np.array([[float(x) for x in line.split()] for line in forces_lines])
        except Exception as e:
            print(f"  [!] Warning: Failed to parse forces from CP2K output: {e}")
    else:
        print("  [!] Warning: Atomic forces not found in CP2K output.")

    return energy, forces

def force_insert_cp2k_output(out_file_path, destination="master"):
    """
    Parses a completed CP2K .out file and directly injects it into either
    the master_train_pool.xyz or training_clean.xyz file, bypassing step checks.
    """
    from mace.calculators import MACECalculator
    from ase import Atoms

    out_path = Path(out_file_path)
    if not out_path.exists():
        print(f"[!] Target CP2K output file does not exist: {out_file_path}")
        return

    print(f"\n[→] Manually parsing {out_path.name} for direct injection into: {destination.upper()}")

    try:
        content = out_path.read_text()
        
        if "SCF run NOT converged" in content:
            print(f"  [✗] Failed to parse: SCF did not converge.")
            return

        # 1. Parse Structure Matrix using your active_pipeline machinery
        cell_matrix = parse_cell_from_out(content)
        symbols, positions = parse_positions_from_out(content)
        
        if cell_matrix is None or symbols is None or positions is None:
            print(f"  [✗] Structural geometries missing from output file.")
            return

        # Create basic Atoms object framework
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell_matrix, pbc=True)

        # 2. Extract Energy using exact float matching
        energy_match = re.search(r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-\d.]+)", content)
        if not energy_match:
            print(f"  [✗] Energy data not found.")
            return
        
        # Note: Replace 'HA_TO_EV' with 'Hartree' if your script imports Hartree from ase.units
        atoms.info["REF_energy"] = float(energy_match.group(1)) * Hartree

        # 3. Extract Forces using your exact column filtering rule (len(parts) == 6)
        force_block = re.search(r"ATOMIC FORCES in \[a\.u\.\](.*?)SUM OF ATOMIC FORCES", content, re.DOTALL)
        forces_ok = False
        if force_block:
            force_lines = force_block.group(1).strip().split("\n")
            forces = []
            for line in force_lines:
                parts = line.split()
                if len(parts) == 6:  # Handles header removal natively!
                    # Note: Replace 'HA_BOHR_TO_EV_ANG' with 'Hartree / Bohr' if using ase.units conversions
                    fx = float(parts[3]) * (Hartree / Bohr)
                    fy = float(parts[4]) * (Hartree / Bohr)
                    fz = float(parts[5]) * (Hartree / Bohr)
                    forces.append([fx, fy, fz])

            if len(forces) == len(atoms):
                atoms.arrays["REF_forces"] = np.array(forces)
                forces_ok = True
            else:
                print(f"  [!] Force row mismatch: read {len(forces)}, expected {len(atoms)} lines.")

        if not forces_ok:
            print(f"  [✗] Failed to validate force matrices. Skipping injection.")
            return

        # 4. Extract Stress Tensor and convert to Voigt order
        stress = parse_stress_from_out(content) if 'parse_stress_from_out' in globals() else None
        if stress is not None:
            voigt = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
            atoms.info["REF_stress"] = voigt

    except Exception as e:
        print(f"  [✗] Error reading or parsing output text matrix: {e}")
        return

    # Extract metadata properties from filename conventions
    stem_match = re.search(r"sp_(.*?)_r\d+_", out_path.stem)
    system_tag = stem_match.group(1) if stem_match else "forced_insertion"
    atoms.info["system_type"] = system_tag
    atoms.info["al_round"] = ROUND
    atoms.info["source"] = "cp2k_sp"

    # Check for exact duplicate geometry entries in target file before appending
    target_file = "master_train_pool.xyz" if destination == "master" else "training_clean.xyz"
    new_hash = get_atoms_hash(atoms)
    
    if Path(target_file).exists():
        try:
            existing_frames = read(target_file, index=":")
            for frame in existing_frames:
                if get_atoms_hash(frame) == new_hash:
                    print(f"  [~] Aborted: This exact configuration is already present in {target_file}.")
                    return
        except:
            pass

    # Handle structural validation layer required specifically for training clean records
    if destination == "clean":
        print("  [→] Aligning MACE baseline residuals for training compatibility...")
        from MACE_CP2K_pipeline.src.active_pipeline import MODEL_PATH
        try:
            calc_mace = MACECalculator(model_paths=MODEL_PATH, device="cuda", default_dtype="float32")
            if APPLY_D3:
                print(f"[→] Including D3 in calculations (MACE + D3)")
                calc_DFT = TorchDFTD3Calculator(
                    device="cuda",
                    damping="bj",
                    xc=cfg.get("dispersion_xc", "pbe"),
                    cutoff=cfg.get("dispersion_cutoff", 40.0),
                )
                calc = SumCalculator([calc_mace, calc_DFT])  
            else:
                print(f"[→] Using MACE only (no D3)")
                calc = calc_mace  
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            atoms.info["MACE_energy"] = atoms_copy.get_potential_energy()
            atoms.array["MACE_forces"] = atoms_copy.get_forces()
        except Exception as e:
            print(f"  [!] Warning: Could not initialize MACE calculator layers: {e}")

    # Append directly to destination file
    try:
        write(target_file, atoms, format="extxyz", append=True)
        print(f"  [✓] Successfully forced and appended structure into {target_file}")
    except Exception as e:
        print(f"  [✗] Failed to write structural frame update onto disk: {e}")


def track_and_recover_structures(file_path, force_cp2k=False, n_fps_frames=3):
    """Audits structural files through active learning stages (unchanged functionality)."""
    path = Path(file_path)
    if not path.exists():
        alt_path = Path("geo_opt_results") / path.name
        if alt_path.exists(): path = alt_path
        else:
            print(f"\n[!] File not found: {file_path}")
            return

    try: all_frames = read(str(path), index=":")
    except Exception as e:
        print(f"[!] Unable to load {path.name}: {e}")
        return

    print("\n" + "="*75)
    print(f"  Auditing Pathway: {path.name} ({len(all_frames)} total frame(s))")
    print("="*75)

    if len(all_frames) == 1:
        frames_to_process = [(0, all_frames[0])]
    else:
        if force_cp2k:
            print(f"[→] Running FPS to sample {min(n_fps_frames, len(all_frames))} images...")
            feats = []
            for a in all_frames:
                pos = a.get_positions().flatten()
                feats.append(pos[:300] if len(pos) >= 300 else np.pad(pos, (0, 300 - len(pos))))
            feats = normalize(np.array(feats))
            selected_indices = [0]
            min_dists = np.linalg.norm(feats - feats[0], axis=1)
            for _ in range(1, min(n_fps_frames, len(all_frames))):
                next_idx = int(np.argmax(min_dists))
                selected_indices.append(next_idx)
                min_dists = np.minimum(min_dists, np.linalg.norm(feats - feats[next_idx], axis=1))
            frames_to_process = [(idx, all_frames[idx]) for idx in selected_indices]
        else:
            frames_to_process = [(len(all_frames)-1, all_frames[-1])]

    for index, atoms_target in frames_to_process:
        target_hash = get_atoms_hash(atoms_target)
        target_stem = path.stem.replace("_initial_opt", "").replace("_final_opt", "")
        if len(all_frames) > 1: target_stem = f"{target_stem}_img{index:02d}"

        stages = {
            "1. Found in AL Candidates Folder (.extxyz)": False,
            "2. CP2K Configuration Built (.inp)": False,
            "3. CP2K Run Evaluated/Succeeded (.out)": False,
            "4. Inserted into Master Train Pool (master_train_pool.xyz)": False,
            "5. Passed Checks into Training Clean (training_clean.xyz)": False,
            "5b. Flagged/Rejected in Outlier Log (training_bad.xyz)": False
        }
        
        # Quick check algorithms matching previous implementations
        al_dir = Path("geo_opt_results/al_candidates")
        if al_dir.exists():
            for al_file in al_dir.glob("*.extxyz"):
                try:
                    for frame in read(str(al_file), index=":"):
                        if get_atoms_hash(frame) == target_hash:
                            stages["1. Found in AL Candidates Folder (.extxyz)"] = True
                            break
                except: pass
                if stages["1. Found in AL Candidates Folder (.extxyz)"]: break

        found_inp = False
        for cp2k_dir in Path(".").glob("cp2k_sp_round*"):
            for inp_file in cp2k_dir.glob(f"*{target_stem}*.inp"):
                found_inp = True
                stages["2. CP2K Configuration Built (.inp)"] = True
                out_file = cp2k_dir / f"{inp_file.stem}.out"
                if out_file.exists():
                    content = out_file.read_text()
                    if "ENERGY| Total FORCE_EVAL" in content and "SCF run NOT converged" not in content:
                        stages["3. CP2K Run Evaluated/Succeeded (.out)"] = True
                        break
            if found_inp: break

        for filename, stage_key in [("master_train_pool.xyz", "4. Inserted into Master Train Pool (master_train_pool.xyz)"),
                                   ("training_clean.xyz", "5. Passed Checks into Training Clean (training_clean.xyz)"),
                                   ("training_bad.xyz", "5b. Flagged/Rejected in Outlier Log (training_bad.xyz)")]:
            if Path(filename).exists():
                try:
                    for frame in read(filename, index=":"):
                        if get_atoms_hash(frame) == target_hash:
                            stages[stage_key] = True
                            break
                except: pass

        print(f"\n  ‣ Sub-frame Status sheets: {target_stem}")
        for stage, passed in stages.items():
            print(f"      {'[✓]' if passed else '[✗]'} {stage}")

        if force_cp2k and not stages["2. CP2K Configuration Built (.inp)"]:
            cp2k_dir = f"cp2k_sp_round{ROUND}"
            os.makedirs(cp2k_dir, exist_ok=True)
            job_name = f"sp_FORCED_{target_stem}_r{ROUND}"
            atoms_target.info["system_type"] = f"FORCED_{target_stem}"
            
            inp_path = write_cp2k_sp(atoms_target, job_name, cp2k_dir)
            forced_jobs = [(job_name, inp_path)]
            script_path = Path(cp2k_dir) / "submit_all.sh"
            missing_path = Path(cp2k_dir) / "submit_missing.sh"

            num_jobs = len(forced_jobs)*n_fps_frames
            
            for script_file in [script_path, missing_path]:
                _write_submission_script(
                    path=script_file, 
                    jobs=forced_jobs, 
                    cp2k_dir=cp2k_dir, 
                    label="Forced Execution", 
                    n_total=num_jobs, 
                    n_skipped=0,
                    append=True
                )
            
            print(f"    [+✓] Forced CP2K input written: {inp_path} for both missing and full submission scripts.")
            print(f"    [+✓] Submission scripts: {script_path} and {missing_path} created for forced execution.")

def resolve_out_files(path_list):
        expanded = []
        for p in path_list:
            path_obj = Path(p)
            if path_obj.is_dir():
                # Recursively finds all .out files inside the folder and sorts them
                expanded.extend([str(f) for f in sorted(path_obj.rglob("*.out"))])
            else:
                expanded.append(p)
        return expanded

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-file/NEB path pipeline tracking script.")
    
    # Core Audit Parameters
    parser.add_argument("--audit", nargs="+", help="List of files to track through active learning pipeline stages.")
    parser.add_argument("--force-cp2k", action="store_true", help="Force generate CP2K inputs for frames that were skipped.")
    parser.add_argument("--n-fps", type=int, default=3, help="Number of distinct images to pull using FPS mapping.")
    
    # NEW: Direct Entry Overrides
    parser.add_argument("--add-master", nargs="+", default=[], help="List of completed CP2K .out paths to parse directly into master_train_pool.xyz")
    parser.add_argument("--add-clean", nargs="+", default=[], help="List of completed CP2K .out paths to parse directly into training_clean.xyz")
    
    args = parser.parse_args()
    print(f"\n[→] Active Learning Pipeline Audit v1.0 | Round {ROUND}\n{'='*75}")

    # Execute Manual Forcing Blocks
    if args.add_master:
        for out_file in args.add_master:
            force_insert_cp2k_output(out_file, destination="master")

    if args.add_clean:
        for out_file in args.add_clean:
            force_insert_cp2k_output(out_file, destination="clean")

    # Execute Auditing Block
    if args.audit:
        print(f"Writing audit report for {len(args.audit)} target file(s)...")
        print(f"\n[→] Auditing with force_cp2k={args.force_cp2k}, n_fps_frames={args.n_fps}\n{'-'*75}")
        print(f"OUTPUT: cp2k_sp_round{ROUND}/submit_all.sh will be generated for any forced CP2K runs.")
        for target_file in args.audit:
            track_and_recover_structures(target_file, force_cp2k=args.force_cp2k, n_fps_frames=args.n_fps)
        