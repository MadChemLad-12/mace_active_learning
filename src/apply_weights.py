#!/usr/bin/env python3
"""
Two-stage per-system_type weighting for MACE training data.

Stage 1  (init):   scan an extxyz, write weights.json listing every system_type,
                   its frame count and any weight already stored in the file.
                   You then edit the "MACE_weight" values by hand.
Stage 2  (apply):  read the filled weights.json, label every frame in the xyz
                   with MACE_weight (a label MACE ignores), set config_type to the
                   group name, and write a one-line JSON for --config_type_weights.
                   The weights reach MACE only through that JSON string.

Usage
-----
  python mace_weights.py init  training.xyz  weights.json
  # ... edit weights.json ...
  python mace_weights.py apply training.xyz  weights.json  training_weighted.xyz

Options
-------
  --key system_type        info key that defines the group (default: system_type)
  --fallback-key config_type   used when a frame has no --key (default: config_type)
  --default-weight 1.0     weight for groups left null in the JSON (apply only);
                           omit to make apply stop instead
  --keep-config-type       do NOT overwrite info["config_type"] with the group name
                           (default is to overwrite, so the JSON keys match MACE)
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

from ase.io import read, write

UNLABELLED = "UNLABELLED"


def group_name(atoms, key, fallback_key):
    name = atoms.info.get(key)
    if name is None:
        name = atoms.info.get(fallback_key)
    return str(name) if name is not None else UNLABELLED


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def init_json(xyz_path, json_path, key, fallback_key):
    frames = read(xyz_path, ":")                      # ":" = all frames
    groups = OrderedDict()                            # name -> {"count", "weights"}
    for atoms in frames:
        name = group_name(atoms, key, fallback_key)
        g = groups.setdefault(name, {"count": 0, "weights": set()})
        g["count"] += 1
        w = atoms.info.get("MACE_weight")
        if w is not None:
            g["weights"].add(float(w))

    # keep anything the user already typed into an existing JSON
    old = {}
    if Path(json_path).exists():
        old = json.load(open(json_path))
        print(f"[init] {json_path} exists - keeping weights you already set")

    out = OrderedDict()
    for name in sorted(groups):
        g = groups[name]
        ws = sorted(g["weights"])
        if len(ws) > 1:
            print(f"[warn] {name}: frames carry different MACE_weight values {ws}")
        current = ws[0] if len(ws) == 1 else (ws if ws else None)
        prior = old.get(name, {}).get("MACE_weight")
        out[name] = {
            "count": g["count"],
            "current_weight": current,                # what the xyz holds now
            # prefill: previous JSON value > current xyz weight > null (fill me in)
            "MACE_weight": prior if prior is not None
            else (current if isinstance(current, float) else None),
        }
        if name == UNLABELLED:
            print(f"[warn] {g['count']} frames have neither '{key}' nor '{fallback_key}'")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    n_missing = sum(v["MACE_weight"] is None for v in out.values())
    print(f"[init] {len(frames)} frames, {len(out)} groups -> {json_path}")
    print(f"[init] {n_missing} groups still have MACE_weight = null; edit them, then run 'apply'")


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def apply_weights(xyz_path, json_path, out_xyz, key, fallback_key,
                  default_weight, set_config_type):
    table = json.load(open(json_path))
    frames = read(xyz_path, ":")

    # fail early, before touching anything
    names = {group_name(a, key, fallback_key) for a in frames}
    unknown = sorted(names - set(table))
    if unknown:
        sys.exit(f"[error] groups in xyz but not in JSON (re-run init): {unknown}")
    null = sorted(n for n in names if table[n].get("MACE_weight") is None)
    if null and default_weight is None:
        sys.exit(f"[error] null MACE_weight for {null}. Fill them in or pass --default-weight")

    weights_out = {}
    stripped = 0
    for atoms in frames:
        name = group_name(atoms, key, fallback_key)
        w = table[name].get("MACE_weight")
        w = float(default_weight if w is None else w)
        if w < 0:
            sys.exit(f"[error] negative weight for {name}")
        atoms.info["MACE_weight"] = w
        # MACE multiplies a per-frame config_weight by the --config_type_weights
        # value, so strip it to avoid double counting. MACE_weight is not a key
        # MACE reads, so it is just a label.
        if "config_weight" in atoms.info:
            atoms.info.pop("config_weight")
            stripped += 1
        if set_config_type:
            atoms.info["config_type"] = name
        weights_out[name] = w

    write(out_xyz, frames, format="extxyz")

    json_out = Path(json_path).with_name(Path(json_path).stem + "_mace_config_type_weights.json")
    with open(json_out, "w") as f:
        json.dump(weights_out, f)                     # one line: safe for $(cat file)

    print(f"[apply] wrote {len(frames)} frames -> {out_xyz}")
    print(f"[apply] wrote {json_out}")
    if stripped:
        print(f"[apply] removed existing config_weight from {stripped} frames (avoids double counting)")
    print(f"[apply] use with:  --config_type_weights \"$(cat {json_out})\"")
    if not set_config_type:
        print("[note] keys are system_type names; they only match MACE's config_type "
              "if the two are identical, otherwise drop --keep-config-type")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("xyz"); pi.add_argument("json")

    pa = sub.add_parser("apply")
    pa.add_argument("xyz"); pa.add_argument("json"); pa.add_argument("out_xyz")
    pa.add_argument("--default-weight", type=float, default=None)
    pa.add_argument("--keep-config-type", action="store_true",
                    help="do not overwrite config_type with the group name")

    for s in (pi, pa):
        s.add_argument("--key", default="system_type")
        s.add_argument("--fallback-key", default="config_type")

    a = p.parse_args()
    if a.cmd == "init":
        init_json(a.xyz, a.json, a.key, a.fallback_key)
    else:
        apply_weights(a.xyz, a.json, a.out_xyz, a.key, a.fallback_key,
                      a.default_weight, not a.keep_config_type)


if __name__ == "__main__":
    main()