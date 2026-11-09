#!/usr/bin/env bash
set -euo pipefail
OUT=FPS_VAL_TEST
MODELS=/home/h.jack/MACE_train/mace_active_learning/model
TEST=fps_validate_frames_corrected.xyz

# Foundation baseline
python analysis/analyse_eval_outputs.py --test_xyz "$TEST" \
  --model "$MODELS/mace-mpa-0-medium.model" --is_foundation --align \
  --outdir "$OUT/mpa0"

for model in "$MODELS"/*.model; do
  # Skip if no matching files found
  [ -e "$model" ] || continue
  
  filename=$(basename "$model")
  
  if [[ "$filename" == "mace-mpa-0-medium.model" ]]; then
    # Skip since it's already evaluated above as the baseline
    continue
  else
    name=$(basename "$model" .model)
    python analysis/analyse_eval_outputs.py --test_xyz "$TEST" \
      --model "$model" \
      --head Default --device cuda \
      --outdir "$OUT/$name"
  fi
done

# 3. Generate summaries, tables, and comparison plots using a Python heredoc
python3 - << 'EOF'
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "FPS_VAL_TEST")
E_THR, F_THR = 10.0, 50.0            # meV/atom, meV/A

def order(p):
    m = re.findall(r"\d+", p.name)
    return int(m[-1]) if m else 0

rows, allf = [], []
for d in sorted([x for x in root.iterdir() if x.is_dir()], key=order):
    sj = next(d.glob("summary_*.json"), None)
    pf = next(d.glob("per_frame_errors_*.csv"), None)
    if not (sj and pf): 
        continue
    
    s = json.load(open(sj))
    df = pd.read_csv(pf)
    df["model"] = d.name
    allf.append(df)
    
    f, e = df.force_rmse_mev, df.energy_error_mev_per_atom.abs()
    rows.append(dict(
        model=d.name, 
        E_rmse=s["energy"]["rmse"], 
        F_rmse=s["forces"]["rmse"],
        F_mae=s["forces"]["mae"], 
        F_median=f.median(), 
        F_p90=f.quantile(0.9),
        F_pass=(f <= F_THR).mean(), 
        E_pass=(e <= E_THR).mean()
    ))

if rows:
    summ = pd.DataFrame(rows).set_index("model")
    print(summ.round(2).to_string())
    summ.to_csv(root / "model_comparison.csv")

    df = pd.concat(allf)
    med = df.pivot_table(index="system_type", columns="model", values="force_rmse_mev",
                         aggfunc="median")[summ.index]
    print("\nMedian frame force RMSE (meV/A) per system_type:\n", med.round(1).to_string())
    med.to_csv(root / "per_system_median_force.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    summ[["F_median", "F_p90", "F_rmse"]].plot(marker="o", ax=ax[0], logy=True)
    ax[0].axhline(F_THR, ls="--", c="k")
    ax[0].set_ylabel("meV/A")
    ax[0].set_title("Force error by version")
    
    med.T.plot(marker="o", ax=ax[1], logy=True, legend=False)
    ax[1].set_title("Median force RMSE per system_type")
    ax[1].legend(fontsize=6, ncol=2)
    
    plt.tight_layout()
    plt.savefig(root / "comparison.png", dpi=200)
    print(f"\nSuccessfully generated reports and plot in {root}/")
else:
    print("No valid evaluation directories found to process.")
EOF