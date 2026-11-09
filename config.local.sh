#!/bin/bash
export MACE_FOUNDATION_MODEL="/home/h.jack/MACE_train/mace_active_learning/mace-mpa-0-medium.model"
export MACE_FINETUNED_MODEL="/home/h.jack/MACE_train/mace_active_learning/mace_V9_active_learning.model"
export MACE_DEFAULT_CSV="/home/h.jack/MACE_train/mace_active_learning/configs/configsV9.csv"
export MACE_NEB_CSV="/home/h.jack/MACE_train/mace_active_learning/configs/configs.csv"
export MACE_NEB_OUTPUT="/home/h.jack/MACE_train/mace_active_learning/neb_comparison"
export MACE_PATH="/home/h.jack/MACE_train/mace_active_learning/"
export MACE_TRAINING_PATH="/home/h.jack/MACE_train/mace_active_learning/training_clean_weightV9.6.xyz"
export EVAL_CONFIGS="/home/h.jack/MACE_train/mace_active_learning/fps_validate_frames_corrected.xyz"
export CP2K_LIBDIR="/home/h.jack/MACE_train/mace_active_learning/data/"

export E0_JSON="/home/h.jack/MACE_train/mace_active_learning/configs/E0s-corrected.json"
export PYTHONWARNINGS="ignore::UserWarning:ast,ignore::DeprecationWarning:torch"


# Singularity-specific
export SIF_PATH="/home/jack/containers/mace_pipeline.sif"
export WORK_DIR="/scratch/jack/runs"
export MODELS_DIR="/scratch/jack/models"
export PYTHONPATH="${MACE_PATH}src:${PYTHONPATH}"


# MACE HYPERPARAMETERS
export LOSS="universal"
export VALIDATION_FRACTION=0.1
export BATCH_SIZE=8
export LR=0.0001
export MAX_EPOCHS=500
export SWA_START=650
export PATIENCE=60
export R_MAX=6.0
export FLOAT_TYPE="float32"

export NUM_SAMPLES_PT=1000   # Materials Project frames to mix in during multi-head training
export USE_HEADS=false

RESTART=False

# Weights
# I think energy 1 and forces 100 is better for start
export FORCES_WEIGHT=100
export ENERGY_WEIGHT=5
export STRESS_WEIGHT=0
# SWA weights
# I think energy 10 and forces 100 is better for swa
export SWA=False
export SWA_LR=0.0001
export FORCES_SWA=100
export ENERGY_SWA=10
export STRESS_SWA=0