#!/bin/bash
# =============================================================================
# MACE Sequential Active Learning Training Script
#
# USAGE:
#   bash train_active_learning.sh
#
# BEFORE RUNNING:
#   1. Set ROUND to the current active learning round number
#   2. Confirm FOUNDATION points to the correct base model
#      - Round 1: use the mace-mp foundation
#      - Round 2+: use the final.model from the previous round
#   3. Confirm EVAL_CONFIGS points to your held-out validation set
#
# WHAT THIS SCRIPT DOES:
#   Step 1 — Pre-flight checks  (files exist, GPU visible)
#   Step 2 — mace_run_train     (fine-tune from foundation or previous round)
#   Step 3 — mace_select_head   (extract the fine-tuned head as a standalone model)
#   Step 4 — mace_eval_configs  (evaluate on held-out validation set)
#   Step 5 — plot_parity.py     (energy + force parity plots)
#   Step 6 — Print next-round instructions
#
# OUTPUT FILES (all stamped with round + timestamp):
#   mace_V{ROUND}_active_learning.model        raw multi-head model from training
#   mace_V{ROUND}_active_learning_final.model  extracted single-head model (use this)
#   results_V{ROUND}_{timestamp}.xyz        evaluation predictions
#   plot_V{ROUND}_{timestamp}_*.png         parity plots
#   train_V{ROUND}_{timestamp}.log          full training stdout/stderr
#   training_summary_{timestamp}.log        high-level progress log
#   training_errors_{timestamp}.log         errors only
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo "[INFO]    $(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$SUMMARY_LOG"
}

log_error() {
    # Usage: log_error CONTEXT STEP "message"
    echo "[ERROR]   $(date '+%Y-%m-%d %H:%M:%S') | $1 | $2 | $3" \
        | tee -a "$ERROR_LOG" | tee -a "$SUMMARY_LOG"
}

log_summary() {
    # Usage: log_summary STATUS "message"
    echo "[SUMMARY] $(date '+%Y-%m-%d %H:%M:%S') | Status: $1 | $2" \
        | tee -a "$SUMMARY_LOG"
}

check_file() {
    # Usage: check_file "path" "CONTEXT" "STEP"
    # Exits the script on failure — use for required files only.
    local file=$1 ctx=$2 step=$3
    if [[ ! -f "$file" ]]; then
        log_error "$ctx" "$step" "Required file not found: $file"
        exit 1
    fi
    if [[ ! -s "$file" ]]; then
        log_error "$ctx" "$step" "File exists but is empty: $file"
        exit 1
    fi
}

section() {
    echo ""
    echo "======================================================================="
    echo "  $*"
    echo "======================================================================="
}

# =============================================================================
# CONFIGURATION — edit these before each round
# =============================================================================
ROUND=""
FOUNDATION=""
TRAINING_PATH=""
# 2. Parse command line arguments (e.g., bash train_active.sh --round 2)
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --round)
            ROUND="$2"
            shift 2 # Past the flag and its value
            ;;
        --foundation)
            FOUNDATION="$2"
            shift 2 # Past the flag and its value
            ;;
        --training)
            TRAINING_PATH="$2"
            shift 2 
            ;;
        -h|--help)
            echo "Usage: $0 --round [number] --foundation [path] --training [path]"
            exit 0
            ;;
        *) # Catch-all for unknown arguments
            echo "Error: Unknown option $1"
            echo "Usage: $0 --round [number] [--foundation path] [--training path]"
            exit 1
            ;;
    esac
done

# 3. Dynamic Foundation Model Selection
# Logic: If Round 1, use base mace-mp. If Round > 1, use 'final.model' from Round-1
if [[ -z "$FOUNDATION" ]]; then
    if [[ "$ROUND" -eq 1 ]]; then
        FOUNDATION="mace-mp-0b3-medium-float32.model"
        log_info "Round 1: Using Foundational MACE-MP model."
    else
        PREV_ROUND=$((ROUND - 1))
        FOUNDATION="mace_V${PREV_ROUND}_active_learning_final.model"
        
        if [ ! -f "$FOUNDATION" ]; then
            log_error "CONFIG" "foundation check" "Previous model $FOUNDATION not found! Did Round $PREV_ROUND finish?"
            exit 1
        fi
        log_info "Round $ROUND: Fine-tuning from previous model: $FOUNDATION"
    fi
else
    log_info "User provided explicit foundation model: $FOUNDATION"
fi

# Training data produced by active_pipeline.py --parse
# Change to master train pool round 2 and onwards.
if [[ -z "$TRAINING_PATH" ]]; then
    TRAINING_PATH="training_clean.xyz"
fi


# Held-out validation set — a fixed set of DFT-labelled frames NOT used in
# training, used to track model quality across rounds.
# If you don't have one yet, point this at master_train_pool.xyz as a proxy
# and create a proper held-out set after round 1.
EVAL_CONFIGS="${EVAL_CONFIGS:-fps_validate_framesV1.xyz}"

# Training hyperparameters
VALIDATION_FRACTION=0.1
BATCH_SIZE=4
LR=0.0001
MAX_EPOCHS=400
SWA_START=300
PATIENCE=30
R_MAX=6.0
NUM_SAMPLES_PT=1200   # Materials Project frames to mix in during multi-head training

# Elements present across ALL your systems (atomic numbers)
# H=1, C=6, O=8, F=9, S=16, Pt=78
# Change to your E0_Values
ATOMIC_NUMBERS="[1, 6, 8, 9, 16, 78]"
E0_VALUES="{
1: -12.6294, 
6: -146.3745, 
8: -431.6014, 
9: -656.5253, 
16: -274.7039, 
78: -3264.7049}"

#E0_VALUES="{
#1: -17.34251075889506, 
#6: -288.66025387600985, 
#8: -434.5100510204964, 
#9: -609.3938692937941, 
#16: -32.07336154177861, 
#78: -3270.692240970651}"

# E0 Values from cp2k single atoms
# E0_VALUES=$(python3 -c "import json; print(json.load(open('E0s.json')))" 2>/dev/null)

# Path to parity plot script
PARITY_SCRIPT="${PARITY_SCRIPT:-plot_parity.py}"

# =============================================================================
# DERIVED PATHS — do not edit
# =============================================================================

timestamp=$(date +%Y%m%d_%H%M)
MODEL_NAME="mace_V${ROUND}_active_learning"
SWA_MODEL="${MODEL_NAME}_stagetwo.model"
FINAL_MODEL="${MODEL_NAME}_final.model"
TRAIN_LOG="train_V${ROUND}_${timestamp}.log"
SUMMARY_LOG="training_summary_${timestamp}.log"
ERROR_LOG="training_errors_${timestamp}.log"
EVAL_OUTPUT="results_V${ROUND}_${timestamp}.xyz"
PLOT_PREFIX="plot_V${ROUND}_${timestamp}"

# =============================================================================
# STEP 1 — Pre-flight checks
# =============================================================================

section "STEP 1 — Pre-flight checks  (Round ${ROUND})"

log_info "===== MACE Training Run Started: $timestamp ====="
log_info "Round            : $ROUND"
log_info "Foundation model : $FOUNDATION"
log_info "Training data    : $TRAINING_PATH"
log_info "Eval configs     : $EVAL_CONFIGS"
log_info "Output model     : $FINAL_MODEL"

check_file "$FOUNDATION"    "PREFLIGHT" "foundation model"
check_file "$TRAINING_PATH" "PREFLIGHT" "training data"

# Warn (don't abort) if eval set is missing — eval is non-fatal
if [[ ! -f "$EVAL_CONFIGS" ]]; then
    log_error "PREFLIGHT" "eval configs" \
        "$EVAL_CONFIGS not found — eval + parity plots will be skipped"
    SKIP_EVAL=true
else
    SKIP_EVAL=false
fi

# Check GPU is visible
if ! nvidia-smi &>/dev/null; then
    log_error "PREFLIGHT" "GPU check" \
        "nvidia-smi failed — training will fall back to CPU (very slow)"
fi

log_info "Pre-flight passed."

# =============================================================================
# STEP 2 — Training
# =============================================================================

section "STEP 2 — mace_run_train"
log_info "Starting training. Full output → $TRAIN_LOG"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mace_run_train \
    --name="$MODEL_NAME" \
    --foundation_model="$FOUNDATION" \
    --train_file="$TRAINING_PATH" \
    --valid_fraction="$VALIDATION_FRACTION" \
    --default_dtype="float32" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --atomic_numbers="$ATOMIC_NUMBERS" \
    --E0s="$E0_VALUES" \
    --model="MACE" \
    --forces_weight=1000 \
    --energy_weight=1 \
    --stress_weight=0 \
    --r_max="$R_MAX" \
    --batch_size="$BATCH_SIZE" \
    --valid_batch_size="$BATCH_SIZE" \
    --lr="$LR" \
    --scheduler="ReduceLROnPlateau" \
    --max_num_epochs="$MAX_EPOCHS" \
    --patience="$PATIENCE" \
    --swa \
    --start_swa "$SWA_START" \
    --swa_energy_weight=10 \
    --swa_forces_weight=500 \
    --swa_stress_weight=0 \
    --multiheads_finetuning=True \
    --pt_train_file="mp" \
    --num_samples_pt="$NUM_SAMPLES_PT" \
    --device=cuda 

train_exit=${PIPESTATUS[0]}

if [[ $train_exit -ne 0 ]]; then
    log_error "TRAINING" "mace_run_train" \
        "Training failed (exit $train_exit). Check $TRAIN_LOG"
    exit 1
fi

check_file "$SWA_MODEL" "TRAINING" "post-training model check"
log_info "Training completed. Model saved: $SWA_MODEL"

# =============================================================================
# STEP 3 — Head extraction
# =============================================================================

section "STEP 3 — mace_select_head"
log_info "Extracting fine-tuned head from $SWA_MODEL → $FINAL_MODEL"

mace_select_head "$SWA_MODEL" \
    --head_name="Default" \
    --output="$FINAL_MODEL" \
    2>&1 | tee -a "$TRAIN_LOG"

head_exit=${PIPESTATUS[0]}

if [[ $head_exit -ne 0 ]]; then
    log_error "SELECT_HEAD" "mace_select_head" \
        "Head extraction failed (exit $head_exit). Falling back to full model: $SWA_MODEL"
    FINAL_MODEL="$SWA_MODEL"
elif [[ ! -s "$FINAL_MODEL" ]]; then
    log_error "SELECT_HEAD" "mace_select_head" \
        "Exit 0 but $FINAL_MODEL is missing or empty. Falling back to: $SWA_MODEL"
    FINAL_MODEL="$SWA_MODEL"
else
    log_info "Head extracted successfully: $FINAL_MODEL"
fi

# =============================================================================
# STEP 4 — Evaluation
# =============================================================================

section "STEP 4 — mace_eval_configs"

if [[ "$SKIP_EVAL" == "true" ]]; then
    log_info "Skipping eval — eval config file was not found at pre-flight."
else
    log_info "Evaluating $FINAL_MODEL on $EVAL_CONFIGS"

    mace_eval_configs \
        --model $FINAL_MODEL \
        --configs $EVAL_CONFIGS \
        --output $EVAL_OUTPUT \
        --default_dtype float32 \
        --batch_size 2 \
        --device cuda \
        2>&1 | tee -a "$TRAIN_LOG"

    eval_exit=${PIPESTATUS[0]}

    if [[ $eval_exit -ne 0 ]]; then
        log_error "EVAL" "mace_eval_configs" \
            "Evaluation failed (exit $eval_exit). Parity plots will be skipped."
        SKIP_PLOTS=true
    elif [[ ! -s "$EVAL_OUTPUT" ]]; then
        log_error "EVAL" "mace_eval_configs" \
            "Eval reported success but $EVAL_OUTPUT is missing or empty."
        SKIP_PLOTS=true
    else
        log_info "Evaluation complete. Results → $EVAL_OUTPUT"
        SKIP_PLOTS=false
    fi
fi

# =============================================================================
# STEP 5 — Parity plots
# =============================================================================

section "STEP 5 — Parity plots"

if [[ "${SKIP_EVAL:-true}" == "true" || "${SKIP_PLOTS:-true}" == "true" ]]; then
    log_info "Skipping parity plots (eval was skipped or failed)."
elif [[ ! -f "$PARITY_SCRIPT" ]]; then
    log_error "PARITY" "plot_parity.py" \
        "Script not found at $PARITY_SCRIPT — skipping plots."
else
    python "$PARITY_SCRIPT" \
        --input "$EVAL_OUTPUT" \
        --prefix "$PLOT_PREFIX" \
        2>&1 | tee -a "$TRAIN_LOG"

    plot_exit=${PIPESTATUS[0]}
    if [[ $plot_exit -ne 0 ]]; then
        log_error "PARITY" "plot_parity.py" \
            "Parity plot script failed (exit $plot_exit). Check $TRAIN_LOG."
    else
        log_info "Parity plots written with prefix: $PLOT_PREFIX"
    fi
fi

# =============================================================================
# STEP 6 — Summary and next-round instructions
# =============================================================================

section "STEP 6 — Round ${ROUND} complete"

log_summary "OK" "Round $ROUND finished. Active model: $FINAL_MODEL"
log_info "===== MACE Training Run Finished: $(date '+%Y%m%d_%H%M') ====="

NEXT_ROUND=$((ROUND + 1))

echo ""
echo "-----------------------------------------------------------------------"
echo "  Round ${ROUND} done. To start Round ${NEXT_ROUND}:"
echo ""
echo "  1. In active_pipeline.py:     set ROUND = ${NEXT_ROUND}"
echo "  2. In mace_geo_opt_base.py:   set ROUND = ${NEXT_ROUND}"
echo "  3. In this script:            set ROUND  = ${NEXT_ROUND}"
echo "                                set FOUNDATION = $FINAL_MODEL"
echo ""
echo "  Then run:"
echo "    python mace_geo_opt_base.py"
echo "    python active_pipeline.py"
echo "    bash   cp2k_sp_round${NEXT_ROUND}/submit_all.sh"
echo "    python active_pipeline.py --parse"
echo "    bash   train_active_learning.sh"
echo "-----------------------------------------------------------------------"
echo ""