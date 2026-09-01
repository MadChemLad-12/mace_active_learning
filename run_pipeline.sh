#!/usr/bin/bash
source "$(dirname "$(realpath "$0")")/config.local.sh"

big_gap() {
    echo -e "\n\n\n\n"
}

track_memory() {
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)" >> gpu_memory.log
        sleep 10
    done
}


# Define the Function
run_training_round() {
    # We only pass ROUND as an argument to keep it explicit
    local ROUND=$1
    
    echo "=== Training for round $ROUND ==="
    # Internal timing/naming
    local "$TIMESTAMP"="$(date +%Y%m%d_%H%M%S)"
    local "$START_TIME"="$(date +%s)"
    [ -z "$FOUNDATION" ] && FOUNDATION="${MACE_FOUNDATION_MODEL}"
    local "$NEW_MODEL"="mace_V${ROUND}_active_learning_final.model"

    echo "-----------------------------------------------------------------------"
    echo "ROUND $ROUND | Geo: $GEO_OPT_RUN | Pipe: $PIPELINE_RUN | CP2K: $CP2K_RUN"
    echo "-----------------------------------------------------------------------"
    echo "Current directory: $(pwd)"
    echo "Will be using the model $FOUNDATION for this run"
    echo "Training will be done in the directory $MACE_PATH"

    NEB_ARGS=(--round "$ROUND" --model "$FOUNDATION")
    [ "$SKIP_NEB" = "True" ] && NEB_ARGS+=(--skip-neb)
    [ "$SKIP_PLM" = "True" ] && NEB_ARGS+=(--skip-plumed)
    [ "$SKIP_AIMD" = "True" ] && NEB_ARGS+=(--skip-aimd)    
    # Step 1: geo-opt + NEB
    if [ "$GEO_OPT_RUN" = "True" ]; then
        echo "Running geometry optimizations and NEB..."
        python "${MACE_PATH}src/neb_geo_run.py" "${NEB_ARGS[@]}"       
        echo "Geo opt and NEB finished"
        big_gap
    fi

    # Step 2: select frames
    PIPE_ARGS=()
    [ -n "$EXCLUDE_KEYWORDS" ] && PIPE_ARGS+=(--exclude "$EXCLUDE_KEYWORDS")

    if [ "$PIPELINE_RUN" = "True" ]; then
        echo "Running active learning pipeline..."
        python "${MACE_PATH}src/active_pipeline.py" \
        "${PIPE_ARGS[@]}" \
        --runs "${RUNS}" \
        --model "${FOUNDATION}" \
        --target "$ROUND"
        big_gap     
    fi

    # Step 3: CP2K jobs
    if [ "$CP2K_RUN" = "True" ]; then
        echo "Running CP2K jobs..."
        bash "${MACE_PATH}cp2k_sp_round${ROUND}/submit_missing.sh"
        
        echo "Moving CP2K restart files..."
        rm -f *.wfn.bak-1 ${MACE_PATH}RESTART 2>/dev/null
        rm -f *.wfn ${MACE_PATH}RESTART 2>/dev/null
        big_gap
    fi

    # Step 4: parse DFT & Build Data
    echo "Parsing DFT results for round $ROUND..."
    python ${MACE_PATH}src/active_pipeline.py \
    --parse-all \
    --target "$ROUND" \
    --model "$FOUNDATION"
    cp master_train_pool.xyz "master_train_pool_${ROUND}_backup.xyz"

    big_gap
    echo "Checking residuals..."
    python ${MACE_PATH}src/check_residuals.py

    # Step 5: Retrain MACE
    big_gap
    python ${MACE_PATH}analysis/compare_models.py \
    --make-held-out

    big_gap
    track_memory &
    TRACK_PID=$!
    if [ "$pre_flight" = "True" ]; then
        echo "Running OOM pre-flight check..."
        if python oom_preflight.py \
            --pool "$TRAINING_PATH" \
            --model "$FOUNDATION" \
            --batch_size 2 --max_count 700; then
            echo "Pre-flight passed! Proceeding to training."
        else
            echo "Pre-flight failed! Skipping active learning retraining." >&2
            exit 1
        fi
    fi

    echo "Retraining MACE model..."
    bash "${MACE_PATH}train_active_learning.sh" \
    --round "$ROUND" \
    --foundation "$FOUNDATION" \
    --training "$TRAINING_PATH"
    kill $TRACK_PID

    # Step 6: Compare
    if [ "$COMPARE_MODELS" = "True" ]; then
        echo comparing models and analyzing results...
        python "${MACE_PATH}analysis/compare_models.py" \
        --outdir comparison_results \
        --test held_out.xyz \
        --models "$FOUNDATION" mace_V*_active_learning.model mace_V*_active_learning_stagetwo.model 
        big_gap
    fi  


    echo Checking the loss function and best performing instances
    python "${MACE_PATH}analysis/plotloss.py" \
    --log "pipeline_$ROUND.log" \
    --head Default \
    --out comparison_results/

    local "$END_TIME"="$(date +%s)"
    local "$TIMETAKEN"="$(( (END_TIME - START_TIME) / 60))"
    echo "Round $ROUND completed in $TIMETAKEN minutes."
}

R=1
EXCLUDE_KEYWORDS=""   # set to "" to disable
GEO_OPT_RUN="True"    # set to True if you want to run geometry optimizations
SKIP_NEB="False"      # set NEB to True if you want to skip NEB calculations
SKIP_AIMD="False"      # set AIMD to True if you want to run AIMD simulations
SKIP_PLM="True" ### Plumed is not working yet, so we skip it for now
        
PIPELINE_RUN="True"   # set to True if you want to generate cp2k inputs

TRAINING_PATH="${MACE_TRAINING_PATH}"
FOUNDATION="${MACE_FOUNDATION_MODEL}"

CP2K_RUN="True"       # set to True if you want to run CP2K calculations
RUNS="1"              # set to the number of cp2k calculations

pre_flight="True"     # set to True if you want to run the OOM pre-flight check before training
COMPARE_MODELS="True" # set to True if you want to compare models after training

echo "Starting Round $R. Logging to pipeline_$R.log"
run_training_round $R 2>&1 | tee "pipeline_$R.log"

exit 0