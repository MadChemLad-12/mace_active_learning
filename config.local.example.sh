#!/bin/bash
# Copy this to config.local.sh and fill in your paths.
# config.local.sh is gitignored — never commit it.

# Foundation model
export MACE_FOUNDATION_MODEL="/path/to/mace-mp-0b3-medium-float32.model"

# Path to FINE_TUNED_MODEL for use in neb_model_compare.py
export MACE_FINETUNED_MODEL="path/to/mace_V*_active_learning_stagetwo.model"

# MACE Model used to fine tune
export MACE_PATH="/path/to/mace_active_learning"

# Path to training data (called training_clean.xyz normally)
export MACE_TRAINING_PATH="training_clean.xyz"
# Path to seperate evaluation data (often data the model has not seen before)
export EVAL_CONFIGS="test.xyz"

# Fine-tuned model (for comparison scripts)
export MACE_FINETUNED_MODEL="/path/to/mace_V4_active_learning_stagetwo.model"

# CSV files
export MACE_DEFAULT_CSV="/path/to/config.csv"
# CSV file to use for neb_model_compare.py
export MACE_NEB_CSV="/path/to/Pt_Diss_Neb_test.csv"

# Output directory for neb_model_compare.py
export MACE_NEB_OUTPUT="/home/user/Documents/Programs/For_GIT/MACE_CP2K_pipeline/neb_comparison"

# Output directory for NEB comparison plots
export MACE_NEB_OUTPUT="neb_comparison"

# Singularity-specific
export SIF_PATH="/home/jack/containers/mace_pipeline.sif"
export WORK_DIR="/scratch/jack/runs"
export MODELS_DIR="/scratch/jack/models"

# MACE HYPERPARAMETERS
export VALIDATION_FRACTION=0.1
export BATCH_SIZE=4
export LR=0.0001
export MAX_EPOCHS=2
export SWA_START=1
export PATIENCE=70
export R_MAX=5.0
export NUM_SAMPLES_PT=0   # Materials Project frames to mix in during multi-head training
export FLOAT_TYPE="float32"

# Weights
export FORCES_WEIGHT=100
export ENERGY_WEIGHT=1
export STRESS_WEIGHT=0
# SWA weights
export FORCES_SWA=100
export ENERGY_SWA=5
export STRESS_SWA=0


