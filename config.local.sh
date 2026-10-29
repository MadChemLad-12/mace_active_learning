#!/bin/bash
export MACE_FOUNDATION_MODEL="/home/h.jack/MACE_train/mace_active_learning/mace-mpa-0-medium.model"
export MACE_FINETUNED_MODEL="/home/h.jack/MACE_train/mace_active_learning/mace_V8_active_learning_stagetwo.model"
export MACE_DEFAULT_CSV="/home/h.jack/MACE_train/mace_active_learning/configs/configsV9.csv"
export MACE_NEB_CSV="/home/h.jack/MACE_train/mace_active_learning/configs/configs.csv"
export MACE_NEB_OUTPUT="/home/h.jack/MACE_train/mace_active_learning/neb_comparison"
export MACE_PATH="/home/h.jack/MACE_train/mace_active_learning/"
export MACE_TRAINING_PATH="/home/h.jack/MACE_train/mace_active_learning/training_clean_weight.xyz"
export EVAL_CONFIGS="/home/h.jack/MACE_train/mace_active_learning/fps_validate_frames_corrected.xyz"
export CP2K_LIBDIR="/home/h.jack/MACE_train/mace_active_learning/data/"

# Singularity-specific
export SIF_PATH="/home/jack/containers/mace_pipeline.sif"
export WORK_DIR="/scratch/jack/runs"
export MODELS_DIR="/scratch/jack/models"
export PYTHONPATH="${MACE_PATH}src:${PYTHONPATH}"


# MACE HYPERPARAMETERS
export VALIDATION_FRACTION=0.1
export BATCH_SIZE=4
export LR=0.0005
export MAX_EPOCHS=600
export SWA_START=400
export PATIENCE=70
export R_MAX=5.0
export NUM_SAMPLES_PT=2000   # Materials Project frames to mix in during multi-head training
export FLOAT_TYPE="float64"

# Weights
# I think energy 1 and forces 100 is better for start
export FORCES_WEIGHT=100
export ENERGY_WEIGHT=5
export STRESS_WEIGHT=0
# SWA weights
# I think energy 10 and forces 100 is better for swa
export FORCES_SWA=100
export ENERGY_SWA=50
export STRESS_SWA=0