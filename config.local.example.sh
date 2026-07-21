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
# Pth to seperate evaluation data (often data the model has not seen before)
export EVAL_CONFIGS="test.xyz"

# Fine-tuned model (for comparison scripts)
export MACE_FINETUNED_MODEL="/path/to/mace_V4_active_learning_stagetwo.model"

# CP2K library data directory
export CP2K_LIBDIR="/path/to/cp2k/data"

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



