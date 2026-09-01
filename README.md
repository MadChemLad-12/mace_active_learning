# MACE Active Learning Pipeline

An automated active learning workflow for fine-tuning [MACE](https://github.com/ACEsuit/mace) machine-learning interatomic potentials using CP2K single-point DFT calculations.

The pipeline generates training data by running geometry optimisations, AIMD and NEB (Nudged Elastic Band) calculations with a given MACE foundational model, selects the most uncertain/diverse frames via FPS, submits them to CP2K for DFT reference data, and retrains the model. Each iteration of this loop is called a **round**.

The general idea of the program was for something that can be self contained on a local PC or run on a HPC system. 

---

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow in Detail](#workflow-in-detail)
  - [Step 0 — Configure your systems](#step-0--configure-your-systems)
  - [Step 1 — Geometry optimisation and NEB](#step-1--geometry-optimisation-and-neb)
  - [Step 2 — Frame selection and CP2K inputs](#step-2--frame-selection-and-cp2k-inputs)
  - [Step 3 — Run CP2K](#step-3--run-cp2k)
  - [Step 4 — Parse DFT results](#step-4--parse-dft-results)
  - [Step 5 — Quality checks](#step-5--quality-checks)
  - [Step 6 — Retrain MACE](#step-6--retrain-mace)
  - [Step 7 — Validate and compare models](#step-7--validate-and-compare-models)
- [Configuration Reference](#configuration-reference)
  - [run_pipeline.sh — master switches](#run_pipelinesh--master-switches)
  - [neb_geo_run.py — geometry and NEB settings](#neb_geo_runpy--geometry-and-neb-settings)
  - [active_pipeline.py — selection and CP2K settings](#active_pipelinepy--selection-and-cp2k-settings)
  - [train_active_learning.sh — training hyperparameters](#train_active_learningsh--training-hyperparameters)
- [E0 Reference Energies](#e0-reference-energies)
- [Adapting to a New Chemical System](#adapting-to-a-new-chemical-system)
- [Utility Scripts](#utility-scripts)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_pipeline.sh                              │
│ (orchestrates all steps; edit the round config block at the bottom) │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
       ┌───────────────────────▼──────────────────────────┐
       │  Step 1: neb_geo_run.py                           │
       │  GeoOpt + NEB +  AIMD sampling                    │
       │  → geo_opt_results/al_candidates/*.extxyz         │
       └───────────────────────┬──────────────────────────┘
                               │
       ┌───────────────────────▼──────────────────────────┐
       │  Step 2: active_pipeline.py                       │
       │  Score frames by MACE force uncertainty           │
       │  FPS diversity selection                          │
       │  Write CP2K single-point inputs                   │
       │  → cp2k_sp_round{N}/submit_missing.sh             │
       └───────────────────────┬──────────────────────────┘
                               │
       ┌───────────────────────▼──────────────────────────┐
       │  Step 3: CP2K (run manually or via SLURM)         │
       │  bash cp2k_sp_round{N}/submit_missing.sh          │
       └───────────────────────┬──────────────────────────┘
                               │
       ┌───────────────────────▼──────────────────────────┐
       │  Step 4: active_pipeline.py --parse-all            │
       │  Parse energies + forces + stress from CP2K outputs│
       │  → master_train_pool.xyz (growing dataset)         │
       └───────────────────────┬──────────────────────────┘
                               │
       ┌───────────────────────▼──────────────────────────┐
       │  Step 5: check_residuals.py                       │
       │  Filter bad frames (high forces, bad cohesive E)  │
       │  → training_clean.xyz                             │
       └───────────────────────┬──────────────────────────┘
                               │
       ┌───────────────────────▼──────────────────────────┐
       │  Step 6: train_active_learning.sh                 │
       │  mace_run_train (multi-head fine-tuning)          │
       │  mace_select_head (extract fine-tuned head)       │
       │  → mace_V{N}_active_learning_final.model          │
       └───────────────────────┬──────────────────────────┘
                               │
       ┌───────────────────────▼──────────────────────────┐
       │  Step 7: compare_models.py + plotloss.py          │
       │  Validate against held-out set                    │
       │  → comparison_results/ (plots + metrics)          │
       └──────────────────────────────────────────────────┘
                      ↑ repeat for next round ↑
```

---

## File Structure

```
mace_active_learning/
├── analysis                      ##   Files for post processing  
│   ├── check_gpu_memory.py       
│   ├── check_neb_index.py        
│   ├── compare_cp2k.py           
│   ├── compare_models.py         
│   ├── list_xyz_name.py          
│   ├── neb_model_compare.py      
│   ├── plotloss.py               
│   └── plot_parity.py            
├── config.local.example.sh       ## A config for bash scripts
├── configs                       ## Files for setting up the python scripts
│   ├── configs.csv               
│   ├── constants.py              
│   ├── E0s.json                  
│   ├── neb_model_compare_config.py 
│   ├── README.md                 
│   └── round_configs             
│       ├── round1_active_pipeline.py 
│       ├── round1_check_residual.py  
│       ├── round1_neb_geo_run.py     
│       └── schema.py                 
├── helpers                           ## Helper functions
│   ├── database_extractor.py         
│   ├── hugface_database.py           
│   └── recalculate_val.py            
├── hpc                               ## Future project
│   └── singularity                   ##[#3](https://github.com/MadChemLad-12/mace_active_learning/issues/3)
│       ├── fugaku_submit.sh          
│       ├── mace_pipeline.def         
│       └── run_pipeline_hpc.sh       
├── LICENSE
├── README.md
├── run_pipeline.sh                 # Bash script that runs the entire pipeline
├── src                             ## Main driving files
│   ├── active_pipeline.py          
│   ├── check_residuals.py          
│   ├── neb_geo_run.py              
│   ├── oom_preflight.py            
│   ├── pipeline_audit.py           
│   └── pool_coverage.py            
├── structures                      ## A location to store *xyz, *cif, etc file. Can be stored anywhere though
└── train_active_learning.sh        # Main script to train MACE using multiheads

```

---

## Installation

### Prerequisites

- Linux with a CUDA-capable GPU (tested on CUDA 13 on a Fedora Linux system)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Mamba
- CP2K compiled with `cp2k.ssmp` (tested using CP2K through conda) available on `$PATH`
- A MACE foundation model (e.g. `mace-mp-0b3-medium-float32.model` from [MACE-MP](https://github.com/ACEsuit/mace-mp) was used in testing)

### 1. Clone the repository

```bash
git clone https://github.com/MadChemLad-12/mace_active_learning
cd MACE_CP2K_pipeline
```

### 2. Create the conda environment

The full pinned environment (recommended for reproducibility):

```bash
conda env create -f mace_env.yaml
conda activate mace_env
```

Or a minimal install using pip on top of an existing PyTorch environment:

```bash
pip install mace-torch ase scikit-learn matplotlib pymatgen
pip install -r pip_requirements.txt
```

### 3. Set environment variables

Add the following to your `~/.bashrc` (or equivalent):
You can quickly do this using source config.local.sh
config.local.sh is a way to quickly set up the program for many enviroments at once
You would use a differnt config.local.sh for each system you want the program to run

Or manually through below

```bash
# Path to your CP2K data files (BASIS_MOLOPT, GTH_POTENTIALS, dftd3.dat)
export CP2K_LIBDIR="/path/to/cp2k/data"

# (Optional) Override the default foundation model path
export MACE_FOUNDATION_MODEL="/path/to/mace-mp-0b3-medium-float32.model"
```
# Follow the config.local.sh file for all the paths

### 4. Download a foundation model

```bash
# From the MACE-MP project — choose float32 for training stability
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model
```
For our inital use on a small GPU system, we used the float32 and medium sized 0b3 model.
Converting to float32 is not necessary but reduces the size on disk.

---

## Quick Start

```bash
# 1. Edit configs.csv to point to your initial and final structures
#    (See "Step 0" below for format details)

# 2. Edit the "configs/", "config/round_configs", and "config.local.sh" files
#    Set your element list, KIND_PARAMS, DEFAULT_CELLS, and etc

# 3. Edit E0s.json with your isolated-atom CP2K energies
#    (or let the pipeline generate them automatically on Round 1)
#    I would recommend doing them manually as they are simple and quick calculations to perform

# 4. Set up your round_config/ parameters for your specific round
#    The "schema.py" file will hold the round of a given round that should reflect the number of "round*_active_pipeline.py"
#    I create new files for each round to keep track of what I have changed

# 5. Edit config.local.sh for your MACE training parameters

# 6. Submit bash run_pipeline.sh to produce your mace model

bash run_pipeline.sh
```

---

## Workflow in Detail

### Step 0 — Configure your systems

Edit **`configs.csv`** to list every initial/final structure pair you want to explore. The `Name` column becomes the `system_type` tag throughout the pipeline.

```csv
Name,initial,final
MySlab_OH,structures/slab_clean.cif,structures/slab_OH.cif
MySlab_O,structures/slab_clean.cif,structures/slab_O.cif
BulkWater,structures/water_box.pdb,
```

- `final` can be left blank for systems where you only want a geometry optimisation (no NEB).
- Paths can be absolute or relative to the working directory.
- The `Name` is used to name all output files, so keep it short and without spaces.

---

### Step 1 — Geometry optimisation and NEB

**Script:** `neb_geo_run.py`  
**Called by:** `run_pipeline.sh` when `GEO_OPT_RUN="True"`

Reads `configs.csv` and for each system:
1. Runs BFGS/FIRE geometry optimisation on both the initial and final structures.
2. (Optional) Runs CI-NEB between optimised endpoints.
3. (Optional, Round ≥ 2) Runs AIMD Langevin MD to sample rare configurations.
4. Writes tagged `.extxyz` files to `geo_opt_results/al_candidates/`:
   - `mace_geoopt_{name}.extxyz` — optimised endpoints
   - `mace_neb_{name}.extxyz` — all NEB images
   - `mace_AIMD_{name}.extxyz` — AIMD frames (if enabled)

**Key settings to edit in `neb_geo_run.py`:**

| Variable | Default | Description |
|---|---|---|
| `MACE_MODEL_PATH` | `"mace-mp-0b3-medium-float32.model"` | Model used for GeoOpt/NEB |
| `FMAX` | `0.05` | Force convergence threshold (eV/Å) |
| `N_IMAGES` | `10` | Number of NEB images |
| `SKIP_NEB` | `False` | Skip NEB (set `True` for Round 1) |
| `SKIP_AIMD` | `True` | Skip AIMD (enable from Round 2+) |
| `AIMD_TARGET` | `initial` | Picks the csv structure to run the AIMD simulation |
| `FIX_BY_HEIGHT` | `False` | Fix atoms below `FIX_HEIGHT_THRESHOLD` |

# There are more values but these are the most important
# See the NebGeoRunConfig() class in "schema" or "round1_neb_geo_run" for more details
---

### Step 2 — Frame selection and CP2K inputs

**Script:** `active_pipeline.py` (for flags see round1_active_pipeline.py)
**Called by:** `run_pipeline.sh` when `PIPELINE_RUN="True"`

1. Loads all `.extxyz` files from `geo_opt_results/al_candidates/`.
2. Re-scores every frame with MACE: frames with high force magnitudes are most uncertain.
3. Applies Farthest-Point Sampling (FPS) to select `N_SELECT_TOTAL` diverse, uncertain frames.
4. Runs basic physical sanity checks on each frame.
5. Writes CP2K single-point input files to `cp2k_sp_round{N}/`.
6. Writes two submission scripts:
   - `submit_all.sh` — every job
   - `submit_missing.sh` — only jobs without a completed `.out`

**Key settings in `active_pipeline.py`:**

```python
# ── Elements ──────────────────────────────────────────────────────────
# Will attempt to read from E0s.json file
# Make sure your E0s.json are written well as they will be used to 
# Identify what elements are present in the code
# ── CP2K KIND parameters per element ──────────────────────────────────
KIND_PARAMS = {
    "H":  ("TZV2P-MOLOPT-GTH",        "GTH-PBE-q1"),
    "O":  ("TZV2P-MOLOPT-GTH",        "GTH-PBE-q6"),
    "Pt": ("DZVP-MOLOPT-SR-GTH",      "GTH-PBE-q18"),
    # Add all elements present in your systems
}
# If not manually listed it will attempt to find 
#  equivalents

# ── Default cell sizes for systems without periodic boundary info ──────
DEFAULT_CELLS = {
    "default":    (20.0, 20.0, 20.0),  # fallback — large vacuum box
    "Mystructure":  (11.1, 9.6,  33.0),  # a, b, c in Angstrom
}

# ── How many frames to submit to CP2K per round ───────────────────────
N_SELECT_TOTAL = 200
# can by selected in the run_pipeline.sh file

# ── Exclude system names containing these keywords ────────────────────
EXCLUDE_SYSTEM_KEYWORDS = []
```

> **Note:** `KIND_PARAMS` is the most common source of errors. Make sure every element in your structures has an entry, or CP2K input files will be incomplete.

---

### Step 3 — Run CP2K

After Step 2, run the generated submission script. How you do this depends on your HPC setup.

**Single workstation:**
```bash
bash cp2k_sp_round1/submit_missing.sh
```
Comes with a built in memory watch dog and timeout for long jobs
Ideally jobs should be small within reason
```
Failed jobs are logged to `cp2k_sp_round{N}/failed_jobs.txt`. After fixing/rerunning them:

```bash
python active_pipeline.py --reparse {N}
```

---

### Step 4 — Parse DFT results

```bash
python active_pipeline.py --parse-all {N}
```

This scans **all** `cp2k_sp_round*/` directories (not just the current round), parses energies and forces from every complete `.out` file, deduplicates by MD5 geometry hash, and appends new frames to `master_train_pool.xyz`.

---

### Step 5 — Quality checks

```bash
python check_residuals.py
```

Filters `master_train_pool.xyz` for:
- **Cohesive energy range** — system-type-aware bounds catch failed SCF convergence disguised as valid output.
- **Maximum reference force** — rejects frames where CP2K forces exceed 25 eV/Å.
- **MACE force RMSE** — rejects frames the current model already predicts well (saves training budget).
- **Geometry checks** — configurable slab atom burial check.

Outputs: `training_clean.xyz` (pass) and `training_bad.xyz` (fail).

NOTE: You may need to edit the "round*_check_residual.py" values manually to ensure they are right for your system
---

### Step 6 — Retrain MACE

**Script:** `train_active_learning.sh`  
**Called by:** `run_pipeline.sh`

> **This is the script you'll edit most between rounds.**

```bash
bash train_active_learning.sh --round 1 --foundation mace-mp-0b3-medium-float32.model
```

Key parameters to set:

| Variable | Description |
|---|---|
| `ATOMIC_NUMBERS` | List of atomic numbers present in your data (will be read from your `E0s.json`), e.g. `"[1, 6, 8, 9, 16, 78]"` |
| `E0_VALUES` | Your CP2K isolated-atom energies generated from `E0s.json` |
| `MAX_EPOCHS` | Training epochs (200 is a good starting point) |
| `SWA_START` | When to switch to SWA/Stage-2 loss (typically ~75% of `MAX_EPOCHS`) |
| `BATCH_SIZE` | Reduce if you hit OOM errors; increase if GPU is underutilised |
| `NUM_SAMPLES_PT` | Materials Project frames for multi-head training (prevents catastrophic forgetting) |

For all the variables see config.local.sh

The script runs four sub-steps automatically:
1. Pre-flight checks (files exist, GPU visible)
2. `mace_run_train` — fine-tune from foundation or previous round
3. `mace_select_head` — extract the fine-tuned head as a standalone model
4. `mace_eval_configs` + `plot_parity.py` — evaluate on held-out set

---

### Step 7 — Validate and compare models

```bash
# Create a held-out set (once, after Round 1)
# Good practice to delete the old held-out.xyz if a large number of structures are added
python compare_models.py --make-held-out

# Compare all rounds against held-out set
python compare_models.py --test held_out.xyz --outdir comparison_results/  --model MACE.model
# You can also use you own data set that the model hasn't seen to check for overtraining
# This is highly recommeded to prevent over training

# Plot training curves
python plotloss.py --log pipeline_1.log --head Default --out comparison_results/
```
Please be mindful as the D3 correctional can introduce a bias that prevents the results from being accurate

`compare_models.py` handles the E0 reference energy shift automatically — it extracts the E0s each model was trained with from the checkpoint, so foundation models (trained with Materials Project E0s) and your fine-tuned models (trained with CP2K E0s) are compared fairly.

---

## Configuration Reference

### `run_pipeline.sh` — master switches

Edit the configuration block near the bottom of the file for each round:

```bash
R=1                          # Round number
GEO_OPT_RUN="True"           # Run neb_geo_run.py?
SKIP_NEB="True"              # Skip NEB? (recommended for Round 1)
SKIP_PLM="True"              # Skip PLUMED? (recommended as it is work in progress)
SKIP_AIMD="TRUE"             # Skip AIMD?
PIPELINE_RUN="True"          # Run active_pipeline.py (frame selection)?
CP2K_RUN="True"              # Run CP2K jobs?
RUNS="150"                   # N_SELECT_TOTAL passed to active_pipeline.py (number of cp2k jobs)
pre_flight="True"     # set to True if you want to run the OOM pre-flight check before training
COMPARE_MODELS="True"        # Run compare_models.py after training?
FOUNDATION="mace-mp-0b3-medium-float32.model"
TRAINING_PATH="training_clean.xyz"
EXCLUDE_KEYWORDS=""          # system_type keywords to drop, e.g. "DRY WET" These system_type are named in the configs.csv file so name them wisely
```

---
### Variables and scripts

### `neb_geo_run.py` — geometry and NEB settings

| Variable | Typical value | Notes |
|---|---|---|
| `MACE_MODEL_PATH` | `"mace-mp-0b3-..."` | Set via `--model` flag or edit directly |
| `FMAX` | `0.05` eV/Å | NEB and GeoOpt convergence |
| `MAX_STEPS` | `500` | Increase for hard systems |
| `OPTIMIZER` | `"FIRE"` | `"BFGS"` is faster on smooth surfaces |
| `N_IMAGES` | `10` | More images = better TS resolution, slower |
| `CLIMB` | `True` | CI-NEB: finds exact transition state |
| `FIX_BY_HEIGHT` | `False` | Fix bottom slab layers |
| `FIX_HEIGHT_THRESHOLD` | `2.7` Å | Atoms below this Z are frozen |
| `APPLY_D3` | `TRUE` | Adds D3 dispersion correction to mace model |
---

### `active_pipeline.py` — selection and CP2K settings

| Variable | Typical value | Notes |
|---|---|---|
| `N_SELECT_TOTAL` | `200` | Frames sent to CP2K per round |
| `FORCE_THRESHOLD` | `0.5` eV/Å | Frames above this are always candidates |
| `MAX_CELL_VOLUME` | `6000` Å³ | Discard oversized cells (saves GPU memory) |
| `REUSE_EXISTING_CP2K` | `True` | Skip already-computed geometries |
| `CP2K_TIMEOUT` | `"4h"` | Per-job timeout string |
| `LIBDIR` | env `CP2K_LIBDIR` | Path to CP2K basis set library |
| `REICON_SAMPLEING` | `TRUE` | Turns on REICO data set generation for a more diverse dataset |
---

### `train_active_learning.sh` — training hyperparameters

| Variable | Typical value | Notes |
|---|---|---|
| `MAX_EPOCHS` | `200` | Increase for later rounds |
| `SWA_START` | `150` | Stage 2 / SWA starts here |
| `PATIENCE` | `50` | Early stopping patience |
| `LR` | `0.0001` | Learning rate |
| `R_MAX` | `6.0` Å | Interaction cutoff |
| `BATCH_SIZE` | `2` | Reduce if GPU OOM |
| `NUM_SAMPLES_PT` | `900` | Materials Project frames for multi-head |
| `VALIDATION_FRACTION` | `0.2` | Fraction of training data held out during training |
| `*_WEIGHTS` `*_SWA` | `number` | Weights for training the mace model for pre and post SWA set |

---

## E0 Reference Energies

`E0s.json` stores the isolated-atom DFT energy for each element (in eV, keyed by atomic number). These are subtracted from total energies to give cohesive/adsorption energies on a consistent scale.

```json
{
  "1":  -12.6294,
  "6":  -146.3745,
  "8":  -431.6014,
  "9":  -656.5253,
  "16": -274.7039,
  "78": -3264.7049
}
```

**To calculate your own E0s** (recommended for a new DFT setup):

```bash
# Generate CP2K inputs for isolated atoms
python active_pipeline.py --e0

# After CP2K runs
python active_pipeline.py --e0 --parse
```

This places each atom in a 20 Å box, runs a CP2K single-point, and writes the results to `E0s.json`.
I generally do this myself without the python script as it is quick and simple to do using a cp2k input.
Make sure this input is the same as what the data is being trained on. 

---

## Adapting to a New Chemical System

To apply this pipeline to a completely different chemistry (e.g. oxide surfaces, zeolites, battery materials), you need to change:

1. **`configs.csv`** — your own initial/final structure files.
2. **`KIND_PARAMS`** in `active_pipeline.py` — CP2K basis sets and pseudopotentials for your elements.
3. **`Z_MAP`** in `active_pipeline.py` — mapping element symbols to atomic numbers.
4. **`DEFAULT_CELLS`** in `active_pipeline.py` — default periodic cell dimensions for your system types.
5. **`E0s.json`** — isolated-atom DFT energies for your elements (or generate them with `--e0`).
6. **`ATOMIC_NUMBERS` and `E0_VALUES`** in `train_active_learning.sh`.
7. **Cohesive energy bounds** in `check_residuals.py` — the `coh_lo/coh_hi` ranges are system-specific.

The slab-burial check in `active_pipeline.py` and `check_residuals.py` is parameterised: set `atoms.info["slab_element"]` and `atoms.info["slab_z_threshold"]` on your structures if you have a slab system.

---

## Utility Scripts
## /analysis
### `check_gpu_memory.py`

Profiles GPU memory usage by running MACE on structures sorted by cell volume. Run this before starting a round to find the maximum structure size your GPU can handle, then set `MAX_CELL_VOLUME` accordingly.

```bash
python check_gpu_memory.py
```

### `check_neb_index.py`

Checks the index and element of the initial and final scrictures from the csv file.
This is to check before running neb_geo_run.py

```bash
python check_neb_index.py
```

### `compare_cp2k.py`

Parses CP2K IT-NEB/CI-NEB output and compares the energy profile
and per-image forces against a MACE MLP model.

```bash
python parse_neb_compare_mace.py \
        --cp2k_out Close0.75PtO2neb.out \
        --replica_dir . \
        --mace_model /path/to/mace_model.model \
        --n_replicas 10
```

### `list_xyz_name.py`

Standalone utility to scan an extended-XYZ (.xyz/.extxyz) trajectory file
and list the unique `system_type=` names found in each frame's header line,
sorted alphabetically (or filtered down to names matching given keywords).

```bash
python list_xyz_name.py path/to/file.xyz
python list_xyz_name.py path/to/file.xyz --keywords dry close
python list_xyz_name.py path/to/file.xyz --keywords retry --case-sensitive
```

### `neb_model_compare.py`

Run NEB simulations for Pt dissolution pathways (Pt, PtO, PtOH, PtO2,
  PtOH2) on solvated Pt(111) at varying O coverages using TWO MACE models
  simultaneously — a foundational model and a fine-tuned model — and compare
  their outputs systematically.

```bash
python neb_model_compare.py [--csv PATH] [--validate-only] [--no-neb]

```

### `plotloss.py`

Parses the multi-head MACE training log and produces figures for each head:
  1. loss_curve_{head}.png     -- loss value vs epoch (Stage 1 and Stage 2)
  2. rmse_curve_{head}.png     -- RMSE_F and RMSE_E vs epoch

```bash
python plotloss.py --log mace_train.log --out my_figures/

```

## /helpers

### `convert_model.py`

converts a mace model to float32 type

```bash
python convert_model.py 'path-to-mace'

```

### `database_extractor.py`

Select the most valuable frames from a candidate database to add to
master_train.xyz, using:

  1) Descriptor-based farthest-point sampling (FPS) for structural diversity
     relative to what's already in master_train.xyz.
  2) (Optional) Committee disagreement across multiple trained MACE models
     to rank by predictive uncertainty.

```bash
python select_frames.py --database candidates.xyz --master master_train.xyz \
    --n-select 200 --no-committee

```

### `hugface_database.py`

Given a specific elemental value, number of structures and the average structure value to accept, will extract structures of value from a given public data set.
The example uses MPtrj and OC25 and builds a 1000 structure data set of specifically structures of chemical interest. 


```bash
python hugface_database.py
    
```

### `recalculate_val.py`

Recalculates DFT single points for a validation xyz using the production
CP2K setup from active_pipeline.py. Results are written to a new XYZ
and NEVER added to the training pool.

```bash
    # Step 1 — write inputs and generate submit script
    python recalculate_validation.py --write

    # Step 2 — run CP2K (on HPC: edit VAL/submit_all.sh first)
    bash VAL/submit_all.sh

    # Step 3 — parse outputs to XYZ
    python recalculate_validation.py --parse

    # Or all steps locally (only if CP2K is available on this machine)
    python recalculate_validation.py --all    
```
## /hpc/singularity
# In the works

## /src

### `pool_coverage.py`

Shows what fraction of your candidate pool (from `neb_geo_run.py`) has already been computed by CP2K, broken down by system type. Helps you decide whether to rerun `neb_geo_run.py` with the new model or continue selecting from the existing pool.

```bash
python pool_coverage.py
```

### `pipeline_audit.py
`

Tracks raw structures or multi-frame NEB pathways through the entire MACE AL 
pipeline to identify precisely where structures fell off or were skipped.

```bash
  # Standard Tracking Audit
  python pipeline_audit.py --audit geo_opt_results/Chair4x4sat_sol_initial_opt.cif
  
  # Force-Parse finished CP2K outputs directly into the master pool
  python pipeline_audit.py --add-master cp2k_sp_round4/sp_Dry0.375PtOH_r4_0004.out
  
  # Force-Parse finished CP2K outputs directly into the clean training pool
  python pipeline_audit.py --add-clean cp2k_sp_round4/sp_Dry0.375PtOH_r4_0015.out```

## Troubleshooting

**`KIND_PARAMS` missing elements → incomplete CP2K input**  
Add the missing element to `KIND_PARAMS` in `active_pipeline.py`. Check CP2K's `BASIS_MOLOPT` and `GTH_POTENTIALS` files for the correct strings.

**OOM during MACE re-scoring**  
Lower `MAX_CELL_VOLUME` in `active_pipeline.py`, or run `check_gpu_memory.py` to find your GPU's limit. Also set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Many frames rejected by `check_residuals.py`**  
Check the rejection reasons in the output. `[cohesive_energy]` rejections often mean your `E0s.json` values are wrong or the CP2K calculation used a different functional/basis. `[high_rmse]` rejections just mean the current model is poor for that system — they will re-enter the pool in a later round.

**NEB fails with atom-count mismatch**  
Your initial and final structures must have identical composition. Check `configs.csv` paths are correct and the PDB/CIF files are consistent.

**CP2K SCF not converging**  
Increase `MAX_SCF` in the `CP2K_TEMPLATE` inside `active_pipeline.py`, or adjust the `ELECTRONIC_TEMPERATURE` (try 300–2000 K for metals). Check the `.out` file for hints.

**`geometry_index_global.json` grows very large**  
This is expected after many rounds. It can be safely deleted and will be rebuilt from the pool on the next run.

---

## Contributing

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear commit messages describing *what* and *why*.
3. If you generalise a system-specific check (e.g. Pt-specific geometry tests), add a comment explaining the generalisation and how users should configure it.
4. Open a pull request with a short description of the change and any relevant test results.

Please open an issue before starting large refactors.


## Notes & Citation

This repository supports my ongoing work on the **Pt–Nafion interface**, studying Pt dissolution 
mechanisms in PEM fuel cells using active-learning MACE machine-learned interatomic potentials (MLIPs).

If you're interested in this type of MACE model or dataset (Pt–Nafion–water interface), feel free to 
reach out — I'm happy to share the data or model from my most accurate iteration.

**A related publication is in preparation and will be linked here once released.**

### Citing this work

If you use this repository in your research, please cite it and reference the associated publication 
(to be added) in your acknowledgments/credits.

### Related work to cite

This pipeline builds on and/or was benchmarked against the following methods and tools. If you use 
this code, please also cite the relevant sources below depending on which components you use:

**MACE**
> Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). 
> MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. 
> *Advances in Neural Information Processing Systems*, 35.
> https://github.com/ACEsuit/mace

# Useful Databases

**OC25**
> Sahoo, S. J., Maraschin, M., Levine, D. S., Ulissi, Z., Zitnick, C. L., Varley, J. B., Gauthier, J. A., 
> Govindarajan, N., & Shuaibi, M. (2025). The Open Catalyst 2025 (OC25) Dataset and Models for 
> Solid-Liquid Interfaces. https://arxiv.org/abs/2509.17862

**CSIRO Platinum Nanoparticle Data Set**
> Barnard, A., Sun, B., & Opletal, G. (2018). Platinum Nanoparticle Data Set. v2. CSIRO. 
> Data Collection. https://doi.org/10.25919/5d3958d9bf5f7

**MPtrj (Materials Project)**
> Horton, M. K., Huck, P., Yang, R. X., Munro, J. M., Dwaraknath, S., Ganose, A. M., Kingsbury, R. S., 
> Wen, M., Shen, J. X., Mathis, T. S., Kaplan, A. D., Berket, K., Riebesell, J., George, J., Rosen, A. S., 
> Spotte-Smith, E. W. C., McDermott, M. J., Cohen, O. A., Dunn, A., Kuner, M. C., Rignanese, G.-M., 
> Petretto, G., Waroquiers, D., Griffin, S. M., … Persson, K. A. (2025). Accelerated data-driven 
> materials science with the Materials Project. *Nature Materials*, 24, 1522–1532. 
> https://www.nature.com/articles/s41563-025-02272-0

# Useful Programs

**Nudged Elastic Band (NEB)**
> Henkelman, G., Uberuaga, B. P., & Jónsson, H. (2000). A climbing image nudged elastic band 
> method for finding saddle points and minimum energy paths. *The Journal of Chemical Physics*, 113(22), 9901–9904.
>
> Henkelman, G., & Jónsson, H. (2000). Improved tangent estimate in the nudged elastic band 
> method for finding minimum energy paths and saddle points. *The Journal of Chemical Physics*, 113(22), 9978–9985.

**ASE (Atomic Simulation Environment)**
> Larsen, A. H., et al. (2017). The atomic simulation environment—a Python library for working 
> with atoms. *Journal of Physics: Condensed Matter*, 29(27), 273002.
> DOI 10.1088/1361-648X/aa680e

**REICO Sampling**
> Yang, C., et al. (2024). Developing General Reactive Element-Based Machine Learning Potentials as the Main Computational Engine for Heterogeneous Catalysis.
> *ChemRxiv*.
> https://doi.org/10.26434/chemrxiv-2024-r8l6j