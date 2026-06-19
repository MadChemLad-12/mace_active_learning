#!/bin/bash
# Copy this to config.local.sh and fill in your paths.
# config.local.sh is gitignored — never commit it.

# Foundation model
export MACE_FOUNDATION_MODEL="/path/to/mace-mp-0b3-medium-float32.model"

# MACE Model used to fine tune
export MACE_PATH="/path/to/mace_active_learning"

# Fine-tuned model (for comparison scripts)
export MACE_FINETUNED_MODEL="/path/to/mace_V4_active_learning_stagetwo.model"

# CP2K library data directory
export CP2K_LIBDIR="/path/to/cp2k/data"

# CSV files
export MACE_DEFAULT_CSV="/path/to/PtDissNeb.csv"
export MACE_NEB_CSV="/path/to/Pt_Diss_Neb_test.csv"

# Output directory for NEB comparison plots
export MACE_NEB_OUTPUT="neb_comparison"

