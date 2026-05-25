#!/usr/bin/bash

#Define base path at top
#Change to your path
export MACE_PATH="/home/user/Documents/Programs/For_GIT/MACE_CP2K_pipeline/"

big_gap() {
    echo -e "\n\n\n\n"
}

### Define the Function
run_training_round() {
    # We only pass ROUND as an argument to keep it explicit
    local ROUND=$1
    
    echo "=== Training for round $ROUND ==="
    # Internal timing/naming
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local START_TIME=$(date +%s)
    # FIX: Only set fallback models if they aren't already explicitly set
    [ -z "$FOUNDATION" ] && FOUNDATION="${MACE_PATH}mace-mp-0b3-medium-float32.model"
    local MACE_OSAKA="${MACE_PATH}mace-osaka24-medium_float32.model"
    local NEW_MODEL="mace_V${ROUND}_active_learning_final.model"

    echo "-----------------------------------------------------------------------"
    echo "ROUND $ROUND | Geo: $GEO_OPT_RUN | Pipe: $PIPELINE_RUN | CP2K: $CP2K_RUN"
    echo "-----------------------------------------------------------------------"
    echo "Current directory: $(pwd)"
    echo "Will be using the model $FOUNDATION for this run"

    NEB_ARGS=(--round "$ROUND" --model "$FOUNDATION")
    [ "$SKIP_OPT" = "True" ] && NEB_ARGS+=(--skip-opt)
    [ "$SKIP_NEB" = "True" ] && NEB_ARGS+=(--skip-neb)
    [ "$SKIP_PLM" = "True" ] && NEB_ARGS+=(--skip-plumed)
    # Step 1: geo-opt + NEB
    if [ "$GEO_OPT_RUN" = "True" ]; then
        echo "Running geometry optimizations and NEB..."
        python "${MACE_PATH}active_learning/neb_geo_run.py" "${NEB_ARGS[@]}"       
        echo "Geo opt and NEB finished"
        big_gap
    fi

    # Step 2: select frames
    PIPE_ARGS=()
    [ "$DISSOLVED" = "True" ] && PIPE_ARGS+=(--dissolve)
    [ -n "$EXCLUDE_KEYWORDS" ] && PIPE_ARGS+=(--exclude $EXCLUDE_KEYWORDS)

    if [ "$PIPELINE_RUN" = "True" ]; then
        echo "Running active learning pipeline..."
        python "${MACE_PATH}active_learning/active_pipeline.py" \
        "${PIPE_ARGS[@]}" \
        --runs "${RUNS}" \
        --model "${FOUNDATION}" \
        "$ROUND"   
        big_gap     
    fi

    # Step 3: CP2K jobs
    if [ "$CP2K_RUN" = "True" ]; then
        echo "Running CP2K jobs..."
        bash "${MACE_PATH}active_learning/cp2k_sp_round${ROUND}/submit_missing.sh"
        
        echo "Moving CP2K restart files..."
        mv *.wfn.bak-1 ${MACE_PATH}active_learning/RESTART 2>/dev/null
        mv *.wfn ${MACE_PATH}active_learning/RESTART 2>/dev/null
        big_gap
    fi

    # Step 4: parse DFT & Build Data
    echo "Parsing DFT results for round $ROUND..."
    python ${MACE_PATH}active_learning/active_pipeline.py --parse-all $ROUND
    cp master_train_pool.xyz "master_train_pool_${ROUND}_backup.xyz"

    echo
    echo "Checking residuals..."
    python ${MACE_PATH}active_learning/check_residuals.py

    # Step 5: Retrain MACE
    echo
    # Bugged atm
    #echo "Checking dataset compusition"
    #python ${MACE_PATH}active_learning/pool_coverage.py

    echo
    echo "Retraining MACE model..."
    bash "${MACE_PATH}active_learning/train_active_learning.sh" \
    --round "$ROUND" \
    --foundation "$FOUNDATION" \
    --training "$TRAINING_PATH"

    # Step 6: Compare
    if [ "$COMPARE_MODELS" = "True" ]; then
        echo comparing models and analyzing results...
        python ${MACE_PATH}active_learning/compare_models_e0s.py --make-held-out
        python ${MACE_PATH}active_learning/compare_models_e0s.py --outdir comparison_results_val --test fps_validate_framesV1.xyz --models $FOUNDATION $MACE_OSAKA mace_V*_active_learning.model mace_V*_active_learning_stagetwo.model V1.1/mace_V1_active_learning_stagetwo.model
        python ${MACE_PATH}active_learning/compare_models_e0s.py --outdir comparison_results --test held_out.xyz --models $FOUNDATION $MACE_OSAKA mace_V*_active_learning.model mace_V*_active_learning_stagetwo.model V1.1/mace_V1_active_learning_stagetwo.model
        big_gap
    fi  


    echo Checking the loss function and best performing instances
    python ../plotloss.py --log pipeline_$ROUND.log --head Default --out comparison_results/

    local END_TIME=$(date +%s)
    local TIMETAKEN=$(( (END_TIME - START_TIME) / 60))
    echo "Round $ROUND completed in $TIMETAKEN minutes."
}

#ONCE THE CURRENT RUN IS DONE. WE DO IT AGAIN
# Run one instance
R=1
EXCLUDE_KEYWORDS=""   # set to "" to disable
GEO_OPT_RUN="True"   
SKIP_OPT="True"
SKIP_NEB="False"
SKIP_PLM="False" ### Plumed isnt really what I was looking for?
        
PIPELINE_RUN="False"
DISSOLVED="False"

TRAINING_PATH="training_clean.xyz"
FOUNDATION="${MACE_PATH}mace-mp-0b3-medium-float32.model"

CP2K_RUN="True"
RUNS="50"
COMPARE_MODELS="True"

echo "Starting Round $R. Logging to pipeline_$R.log"
run_training_round $R 2>&1 | tee "pipeline_$R.log"

exit 0

### Main Execution Loop
for R in {3..4}; do
    # --- SETUP CONFIGURATION FOR THIS ROUND ---
    if [ "$R" -eq 3 ]; then
        GEO_OPT_RUN="True"   # Turn off for Round 1
        SKIP_OPT="False"
        SKIP_NEB="True"
        SKIP_PLM="True"
        
        PIPELINE_RUN="True"
        DISSOLVED="True"

        TRAINING_PATH=""
        FOUNDATION="${MACE_PATH}mace-mp-0b3-medium-float32.model"

        CP2K_RUN="True"
        RUNS="150"
        COMPARE_MODELS="True"

    elif [ "$R" -eq 4 ]; then
        GEO_OPT_RUN="True"   # Turn on for Round 2
        SKIP_OPT="False"
        SKIP_NEB="False"
        SKIP_PLM="False"

        PIPELINE_RUN="True"
        DISSOLVED="False"
        
        TRAINING_PATH=""
        FOUNDATION=""

        CP2K_RUN="True"
        RUNS="200"
        COMPARE_MODELS="False"

    else
        GEO_OPT_RUN="False"    # Turn on for Round 3 and above
        SKIP_OPT="True"
        SKIP_NEB="True"
        SKIP_PLM="True"

        PIPELINE_RUN="True"
        DISSOLVED="True"
        
        TRAINING_PATH=""
        FOUNDATION=""

        CP2K_RUN="True"
        RUNS="60"
        COMPARE_MODELS="True"
    fi
    # ------------------------------------------
    echo "Starting Round $R. Logging to pipeline_$R.log"
    run_training_round $R 2>&1 | tee "pipeline_$R.log"
done
echo "All finished"
