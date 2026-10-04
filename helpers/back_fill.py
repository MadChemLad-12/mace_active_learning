"""
One-off backfill: register every sp_*.inp in cp2k_sp_round1/ into
model_label_index.json, so parse_all_cp2k_outputs() recognizes files that
were written before the model-labeling feature existed.

Registers from .inp files (not just .out), so jobs still awaiting
completion also get an entry — parse_all_cp2k_outputs will pick their
.out up automatically once it exists, no need to re-run this later.

Usage:
    python backfill_round1_labels.py
    python backfill_round1_labels.py --dir cp2k_sp_round1 --model mace-mp-0b3-medium-float32
"""
import argparse
import json
from pathlib import Path

MODEL_LABEL_INDEX = "model_label_index.json"


def load_index():
    p = Path(MODEL_LABEL_INDEX)
    return json.loads(p.read_text()) if p.exists() else {}


def save_index(index):
    Path(MODEL_LABEL_INDEX).write_text(json.dumps(index, indent=2))


def backfill(cp2k_dir: str, model_name: str, overwrite: bool):
    cp2k_dir = Path(cp2k_dir)
    if not cp2k_dir.exists():
        raise SystemExit(f"[!] {cp2k_dir} does not exist")

    index = load_index()
    stems = sorted({p.stem for p in cp2k_dir.glob("sp_*.inp")})

    added, skipped = 0, 0
    for stem in stems:
        if stem in index and not overwrite:
            skipped += 1
            continue
        index[stem] = {"model": model_name, "cp2k_dir": str(cp2k_dir)}
        added += 1

    save_index(index)
    n_out = len(list(cp2k_dir.glob("sp_*.out")))
    print(f"[✓] {cp2k_dir}: {len(stems)} job(s) found ({n_out} completed .out so far)")
    print(f"    {added} entries added, {skipped} already present and left untouched")
    print(f"    → {MODEL_LABEL_INDEX} now has {len(index)} total entries")
    if skipped and not overwrite:
        print(f"    (use --overwrite to relabel existing entries)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="cp2k_sp_round1", help="CP2K round directory to backfill")
    ap.add_argument("--model", default="mace-mp-0b3-medium-float32",
                     help="Model name to register (no .model extension — matches Path(...).stem)")
    ap.add_argument("--overwrite", action="store_true",
                     help="Relabel stems that already have an index entry")
    args = ap.parse_args()

    backfill(args.dir, args.model, args.overwrite)