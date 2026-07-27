#!/usr/bin/env python3
"""
list_structures.py

Standalone utility to scan an extended-XYZ (.xyz/.extxyz) trajectory file
and list the unique `system_type=` names found in each frame's header line,
sorted alphabetically (or filtered down to names matching given keywords).

Parses the file manually, frame-by-frame, without loading atomic positions
into memory — so it stays fast even on very large pool/trajectory files.

Example header line this parses:
    240
    Lattice="11.099 0.0 0.0 0.0 9.612 0.0 0.0 0.0 33.0" \
    Properties=species:S:1:pos:R:3:REF_forces:R:3 system_type=Close0.0Pt \
    REF_energy=-247770.4258588088 al_round=1 source=cp2k_sp \
    stress_missing=T _cp2k_dir=cp2k_sp_round1 pbc="T T T"

Usage:
    python list_structures.py path/to/file.xyz
    python list_structures.py path/to/file.xyz --keywords dry close
    python list_structures.py path/to/file.xyz --keywords retry --case-sensitive
"""

import argparse
import re
import sys
from pathlib import Path


def list_unique_structure_names(xyz_path, keywords=None, case_sensitive=False):
    """
    Scan an extended-XYZ file and return the unique `system_type=` values
    found in each frame's comment/header line, sorted alphabetically.

    Args:
        xyz_path (str): Path to the .xyz / .extxyz file.
        keywords (list[str], optional): If given, only return names that
            contain at least one of these substrings (e.g. ["Close", "Dry"]).
        case_sensitive (bool): Whether keyword matching is case-sensitive.
            Default False (e.g. "dry" matches "Dry0.0Pt").

    Returns:
        list[str]: Unique structure names, alphabetically sorted.
    """
    name_pattern = re.compile(r'system_type=("[^"]*"|\S+)')
    names = set()

    with open(xyz_path) as f:
        while True:
            count_line = f.readline()
            if not count_line:
                break  # EOF
            count_line = count_line.strip()
            if not count_line:
                continue
            try:
                n_atoms = int(count_line)
            except ValueError:
                # Not a valid frame-count line — skip and keep scanning
                continue

            header_line = f.readline()
            match = name_pattern.search(header_line)
            if match:
                names.add(match.group(1).strip('"'))

            # Skip the n_atoms coordinate lines belonging to this frame
            for _ in range(n_atoms):
                if not f.readline():
                    break  # File ended unexpectedly mid-frame

    if keywords:
        if case_sensitive:
            names = {n for n in names if any(kw in n for kw in keywords)}
        else:
            kws_lower = [kw.lower() for kw in keywords]
            names = {n for n in names if any(kw in n.lower() for kw in kws_lower)}

    return sorted(names)


def main():
    parser = argparse.ArgumentParser(
        description="List unique system_type structure names from an extxyz file."
    )
    parser.add_argument("xyz_path", type=str, help="Path to the .xyz/.extxyz file")
    parser.add_argument(
        "--keywords", nargs="*", default=None, metavar="KEYWORD",
        help="Only show names containing any of these substrings (default: case-insensitive)"
    )
    parser.add_argument(
        "--case-sensitive", action="store_true",
        help="Make --keywords matching case-sensitive"
    )
    args = parser.parse_args()

    if not Path(args.xyz_path).exists():
        print(f"[!] File not found: {args.xyz_path}", file=sys.stderr)
        sys.exit(1)

    names = list_unique_structure_names(
        args.xyz_path, keywords=args.keywords, case_sensitive=args.case_sensitive
    )

    label = f" (filtered by keywords: {args.keywords})" if args.keywords else ""
    print(f"[→] {len(names)} unique structure name(s) in {args.xyz_path}{label}:")
    for n in names:
        print(f"    {n}")


if __name__ == "__main__":
    main()