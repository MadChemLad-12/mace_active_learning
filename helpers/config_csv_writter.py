import csv
import os
from typing import List, Dict
import glob
import argparse

"""
This script gets all csv files wihtin a pathed directory and writes all of them into a single csv file.
The csv file will have the headers of "Name,initial,final,charge,multiplicity,fragments"
As an example, the output will look like this:

"Pt100,"/home/h.jack/MACE_train/mace_active_learning/structures/pt100_monolayer.cif",,0,guess,"

By defult the values for final, charge, multiplicity, and fragments will be 0, guess, and "" respectively.

"""

# Supported structure file extensions/suffixes
STRUCTURE_EXTENSIONS = ('.cif', '.xyz', '_POSCAR')


def collect_structure_metadata(directory_path: str) -> List[Dict[str, str]]:
    """Iterates through files in the directory and builds metadata dictionaries."""
    data = []
    
    # Sort files to guarantee consistent output order across OS platforms
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(STRUCTURE_EXTENSIONS):
            file_path = os.path.abspath(os.path.join(directory_path, filename))
            
            # Use filename without extension for the 'Name' column
            name = os.path.splitext(filename)[0]
            
            row = {
                "Name": name,
                "initial": file_path,
                "final": "",
                "charge": "0",
                "multiplicity": "guess",
                "fragments": ""
            }
            data.append(row)
            
    return data

# Function to write the csv file
def write_csv_file(output_file_path: str, data: List[Dict[str, str]]):
    """Writes metadata list into a formatted CSV file."""
    import csv
    
    fieldnames = ["Name", "initial", "final", "charge", "multiplicity", "fragments"]
    
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(
        description='Collect structure file paths (.cif, .xyz, _POSCAR) into a structured CSV file.'
    )
    parser.add_argument('directory', type=str, help='Path to the directory containing structure files.')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file.')
    args = parser.parse_args()

    # Collect structure metadata
    structure_data = collect_structure_metadata(args.directory)

    # Write data to the destination CSV
    write_csv_file(args.output_file, structure_data)
    print(f"Successfully processed {len(structure_data)} structure file(s) -> {args.output_file}")


if __name__ == '__main__':
    main()