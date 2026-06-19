#!/bin/bash
source "$(dirname "$(realpath "$0")")/../config.local.sh"

# The bind mounts: host path → container path
# Since your scripts use absolute paths from env vars, those paths must
# exist inside the container too — easiest way is to bind them to themselves

BINDS=(
    "--bind ${WORK_DIR}:${WORK_DIR}"       # same path inside and outside
    "--bind ${MODELS_DIR}:${MODELS_DIR}"   # so env vars need no changes
    "--bind ${MACE_PATH}:${MACE_PATH}"
)

singularity exec --nv \
    "${BINDS[@]}" \
    "${SIF_PATH}" \
    bash "${MACE_PATH}run_pipeline.sh"