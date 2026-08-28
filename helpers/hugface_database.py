import shutil
from pathlib import Path
from ase.io import read, write
from fairchem.core.datasets import AseDBDataset

# --- GLOBAL CONFIGURATION ---
ELEMENT_VALUES = {
    "Pt": 5.0,
    "C": 2.0,
    "O": 2.0,
    "S": 1.5,
    "F": 1.5,
    "H": 1.0,
}
ELEMENTS_OF_INTEREST = set(ELEMENT_VALUES.keys())
STRUCTURE_COUNT_TARGET = 1000
MIN_AVERAGE_STRUCTURE_VALUE = 2.0

# Define your paths and details for both datasets
DATASET_CONFIGS = [
    
    {"name": "MPtrj (Plain XYZ)",
        "repo_path": Path(
            "/home/user/Documents/Programs/Python/ASE/MACE/hugface_data/training_data/mptrj-gga-ggapu/"
        ),
        "master_output": Path(
            "/home/user/Documents/Programs/Python/ASE/MACE/hugface_data/training_data/mptrj-gga-ggapu/master_ranked_MPtrj_structures.extxyz"
        ),
        "file_extension": "*.extxyz",
        "is_lmdb": False,
    },
    {
        "name": "OC25 (LMDB)",
        "repo_path": Path(
            "/home/user/Documents/Programs/Python/ASE/MACE/hugface_data/train"
        ),
        "master_output": Path(
            "/home/user/Documents/Programs/Python/ASE/MACE/hugface_data/train/master_ranked_OC25_structures.extxyz"
        ),
        "file_extension": "*.aselmdb",
        "is_lmdb": True,
    },
    
]


# --- HELPER FUNCTIONS ---
def is_structure_valuable(atoms):
    """Calculates if the structure's elements make it valuable enough to keep."""
    symbols = atoms.get_chemical_symbols()
    if not symbols:
        return False
    total_value = sum(ELEMENT_VALUES.get(sym, 0.0) for sym in symbols)
    return (total_value / len(symbols)) >= MIN_AVERAGE_STRUCTURE_VALUE


def parse_and_filter_shard(
    file_path, output_xyz, current_count, target_count, is_lmdb
):
    """Reads a file chunk using either LMDB or plain XYZ methods and filters atoms."""
    filtered_structures = []
    structures_needed = target_count - current_count

    # Branching reader logic based on dataset type
    if is_lmdb:
        dataset = AseDBDataset({"src": str(file_path)})
        total_structures = len(dataset)
        print(f"  Total structures in shard: {total_structures}")

        for i in range(total_structures):
            if len(filtered_structures) >= structures_needed:
                break
            atoms = dataset.get_atoms(i)
            symbols = set(atoms.get_chemical_symbols())
            if "Pt" in symbols and symbols.issubset(ELEMENTS_OF_INTEREST):
                if is_structure_valuable(atoms):
                    filtered_structures.append(atoms)
    else:
        try:
            structures_in_file = read(file_path, index=":")
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")
            return []

        print(f"  Total structures in file: {len(structures_in_file)}")
        for atoms in structures_in_file:
            if len(filtered_structures) >= structures_needed:
                break
            symbols = set(atoms.get_chemical_symbols())
            if "Pt" in symbols and symbols.issubset(ELEMENTS_OF_INTEREST):
                if is_structure_valuable(atoms):
                    filtered_structures.append(atoms)

    print(f"  Kept {len(filtered_structures)} structures from this file.")
    if filtered_structures:
        write(output_xyz, filtered_structures, format="extxyz")

    return filtered_structures


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    for config in DATASET_CONFIGS:
        print(f"\n==========================================")
        print(f"STARTING PROCESS FOR: {config['name']}")
        print(f"==========================================")

        repo_path = config["repo_path"]
        master_output_path = config["master_output"]

        created_files = []
        total_structures_collected = 0

        # Gather file targets and filter out pre-existing master files
        all_files = sorted(repo_path.glob(config["file_extension"]))
        files_to_process = [
            f for f in all_files if f.name != master_output_path.name
        ]

        for file_path in files_to_process:
            # Skip any leftovers from aborted runs
            if file_path.name.startswith("temp_filtered_"):
                continue

            print(f"\n--- Processing {file_path.name} ---")
            print(
                f"  Progress: {total_structures_collected} / {STRUCTURE_COUNT_TARGET}"
            )

            # Generate temporary file names to step around infinite loops
            output_xyz = (
                repo_path / f"temp_filtered_{file_path.stem}.tmp_xyz"
            )

            shard_structures = parse_and_filter_shard(
                file_path,
                output_xyz,
                total_structures_collected,
                STRUCTURE_COUNT_TARGET,
                config["is_lmdb"],
            )

            if shard_structures:
                created_files.append(output_xyz)
                total_structures_collected += len(shard_structures)

            if total_structures_collected >= STRUCTURE_COUNT_TARGET:
                print(
                    f"\nTarget achieved! Collected {total_structures_collected} structures."
                )
                break

        # Combine step per dataset loop
        if created_files:
            print(
                f"\n--- Combining {len(created_files)} parts into master file ---"
            )
            with open(master_output_path, "wb") as master_file:
                for f_path in created_files:
                    with open(f_path, "rb") as individual_file:
                        shutil.copyfileobj(individual_file, master_file)

            print(
                f"Success! Final master created at: {master_output_path}"
            )

            # Clean temporary parts up immediately
            for f_path in created_files:
                f_path.unlink()
        else:
            print(f"\nNo matching structures found for {config['name']}.")
            
    ### FINAL VERIFICATION STEP ###
    from collections import Counter
    from pathlib import Path
    from ase.io import read    
    verification_targets = [
        {
            "name": "OC25 Master File",
            "path": Path(
                "/home/user/Documents/Programs/Python/ASE/MACE/hugface_data/train/master_ranked_OC25_structures.extxyz"
            ),
        },
        {
            "name": "MPtrj Master File",
            "path": Path(
                "/home/user/Documents/Programs/Python/ASE/MACE/hugface_data/training_data/mptrj-gga-ggapu/master_ranked_MPtrj_structures.extxyz"
            ),
        },
    ]

    print("\n==========================================")
    print("FINAL MASTER FILE VERIFICATION")
    print("==========================================")

    for target in verification_targets:
        file_path = target["path"]
        print(f"\nChecking {target['name']}...")

        if not file_path.exists():
            print(f"  ❌ File not found at: {file_path}")
            continue

        # 1. Fast structure count via line parsing
        # Extended XYZ files always start a frame with the total number of atoms on its own line
        structure_count = 0
        try:
            with open(file_path, "r") as f:
                for line in f:
                    # If a line contains only a number, it marks the start of a new frame
                    if line.strip().isdigit():
                        structure_count += 1
        except Exception as e:
            print(f"  ❌ Error reading file for counting: {e}")
            continue

        print(f"  ✅ Total Structures Found: {structure_count}")

        # 2. Element tracking via ASE
        try:
            print("  Analyzing elemental composition...")
            frames = read(file_path, index=":")

            all_elements = Counter()
            for atoms in frames:
                # Update counter with the unique elements present in this specific frame
                unique_in_frame = set(atoms.get_chemical_symbols())
                all_elements.update(unique_in_frame)

            print("  ✅ Elements Present (and how many structures they appear in):")
            for element, count in sorted(all_elements.items()):
                print(f"     - {element}: present in {count}/{structure_count} structures")

        except Exception as e:
            print(f"  ❌ Error reading file via ASE for element check: {e}")

    print("\n==========================================")
    print("VERIFICATION COMPLETE")
    print("==========================================")