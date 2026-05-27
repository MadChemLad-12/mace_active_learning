"""
check_structure_pairs.py
========================
Reads the same CSV used by neb_geo_run.py and checks every initial/final
structure pair for index and element consistency.

Checks performed per pair:
  1. File exists on disk
  2. Atom counts match
  3. Element symbols match at every index
  4. Chemical formula matches (catch-all for count differences)

No MACE model is loaded — this is a pure structure check, runs in seconds.

USAGE
-----
  python check_structure_pairs.py
  python check_structure_pairs.py --csv /path/to/other.csv
  python check_structure_pairs.py --fix   # suggest proximity-mapping for mismatches
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase.io import read

# ── Default CSV path — same as neb_geo_run.py ────────────────────────────────
DEFAULT_CSV = "/home/user/Documents/Programs/Python/ASE/MACE/active_learning/PtDIssNeb.csv"

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}[✓]{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[!]{RESET} {msg}")
def err(msg):  print(f"  {RED}[✗]{RESET} {msg}")
def info(msg): print(f"  {CYAN}[→]{RESET} {msg}")


def check_pair(name, init_path, final_path, suggest_fix=False):
    """
    Run all checks for one config pair.
    Returns: (passed: bool, issues: list[str])
    """
    issues = []

    # ── 1. File existence ────────────────────────────────────────────────────
    init_missing  = not Path(init_path).exists()
    final_missing = not Path(final_path).exists()

    if init_missing:
        issues.append(f"Initial file not found: {init_path}")
    if final_missing:
        issues.append(f"Final file not found:   {final_path}")
    if init_missing or final_missing:
        return False, issues

    # ── 2. Load structures ───────────────────────────────────────────────────
    try:
        init_atoms  = read(init_path)
    except Exception as e:
        issues.append(f"Could not read initial file: {e}")
        return False, issues

    try:
        final_atoms = read(final_path)
    except Exception as e:
        issues.append(f"Could not read final file: {e}")
        return False, issues

    n_init  = len(init_atoms)
    n_final = len(final_atoms)

    # ── 3. Atom count ────────────────────────────────────────────────────────
    if n_init != n_final:
        issues.append(
            f"Atom count mismatch — initial: {n_init}, final: {n_final}"
        )
        return False, issues   # no point checking indices if counts differ

    # ── 4. Formula match ─────────────────────────────────────────────────────
    init_formula  = init_atoms.get_chemical_formula()
    final_formula = final_atoms.get_chemical_formula()
    if init_formula != final_formula:
        issues.append(
            f"Formula mismatch — initial: {init_formula}, final: {final_formula}"
        )
        # Continue — still report per-index mismatches below

    # ── 5. Per-index element check ───────────────────────────────────────────
    init_syms  = init_atoms.get_chemical_symbols()
    final_syms = final_atoms.get_chemical_symbols()

    mismatches = [
        (i, s1, s2)
        for i, (s1, s2) in enumerate(zip(init_syms, final_syms))
        if s1 != s2
    ]

    if mismatches:
        issues.append(
            f"Element mismatch at {len(mismatches)} index(es):"
        )
        for i, s1, s2 in mismatches[:10]:
            issues.append(f"    index {i:>5d} — initial: {s1:<4s}  final: {s2}")
        if len(mismatches) > 10:
            issues.append(f"    ... and {len(mismatches) - 10} more")

        if suggest_fix:
            issues.append(
                "Tip: run neb_workflow(..., use_proximity_mapping=True) "
                "to auto-reorder the final structure."
            )

    return len(issues) == 0, issues


def summarise(results):
    """Print the final summary table."""
    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    print("\n" + "=" * 60)
    print(f"{BOLD}SUMMARY{RESET}")
    print("=" * 60)
    print(f"  Total configs : {len(results)}")
    print(f"  {GREEN}Passed{RESET}        : {len(passed)}")
    print(f"  {RED}Failed{RESET}        : {len(failed)}")

    if failed:
        print(f"\n{RED}Configs needing attention:{RESET}")
        for r in failed:
            print(f"  • {r['name']}")
        print(
            "\nRun with --verbose to see per-index details for all configs,\n"
            "or --fix to see proximity-mapping suggestions."
        )
        return 1   # non-zero exit for scripting
    else:
        print(f"\n{GREEN}All pairs are consistent — safe to run NEB.{RESET}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check initial/final structure pairs from the NEB CSV."
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Path to the CSV file (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Suggest proximity-mapping fix for mismatched pairs"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-index details even for passing pairs"
    )
    args = parser.parse_args()

    # ── Load CSV ─────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"{RED}[✗] CSV not found: {args.csv}{RESET}")
        sys.exit(1)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        configs = [
            {"name": row["Name"], "initial": row["initial"], "final": row["final"]}
            for row in reader
        ]

    if not configs:
        print(f"{YELLOW}[!] No configurations found in {args.csv}{RESET}")
        sys.exit(0)

    print(f"\n{BOLD}Structure pair checker{RESET}")
    print(f"CSV: {args.csv}")
    print(f"Configs to check: {len(configs)}\n")
    print("=" * 60)

    # ── Check each pair ───────────────────────────────────────────────────────
    results = []
    for cfg in configs:
        name       = cfg["name"]
        init_path  = cfg["initial"]
        final_path = cfg["final"]

        print(f"\n{BOLD}{name}{RESET}")
        info(f"Initial: {init_path}")
        info(f"Final:   {final_path}")

        passed, issues = check_pair(name, init_path, final_path,
                                    suggest_fix=args.fix)

        if passed:
            ok("All checks passed — indices and elements match.")
            if args.verbose:
                atoms = read(init_path)
                syms  = atoms.get_chemical_symbols()
                formula = atoms.get_chemical_formula()
                info(f"{len(atoms)} atoms | formula: {formula}")
                # Show element composition
                unique = sorted(set(syms))
                counts = {el: syms.count(el) for el in unique}
                info("Composition: " + "  ".join(
                    f"{el}: {counts[el]}" for el in unique
                ))
        else:
            for issue in issues:
                if issue.startswith("    "):   # sub-item, indent only
                    print(f"       {issue}")
                else:
                    err(issue)

        results.append({"name": name, "ok": passed, "issues": issues})

    sys.exit(summarise(results))


if __name__ == "__main__":
    main()