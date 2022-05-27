#!/bin/bash

# CLI parameters
POP_SIZE=${1}
FITNESS_FUNCTION=${2}
EVALTIME_FUNCTION=${3}
EXPERIMENT_NAME=${4}
echo "Running with POP_SIZE=${POP_SIZE} on ${2} and eval-time of ${3}"

# Fixed parameters
JOBS=100
BIRTHS=5000
OUTPUT_DIR=2022-03-07_sweet/
mkdir -p ${OUTPUT_DIR}

echo "Running immediate without SWEET..."
python ../../../examples/async_simulation.py \
    --no-viz \
    --immediate \
    --pop-size=${POP_SIZE} \
    --jobs=${JOBS} \
    --births=${BIRTHS} \
    --fitness-function=${FITNESS_FUNCTION} \
    --evaltime-function=${EVALTIME_FUNCTION} \
    > ${OUTPUT_DIR}/immediate_${EXPERIMENT_NAME}_popsize${POP_SIZE}.csv

echo "Running immediate with SWEET..."
python ../../../examples/async_simulation.py \
    --no-viz \
    --immediate \
    --sweet \
    --pop-size=${POP_SIZE} \
    --jobs=${JOBS} \
    --births=${BIRTHS} \
    --fitness-function=${FITNESS_FUNCTION} \
    --evaltime-function=${EVALTIME_FUNCTION} \
    > ${OUTPUT_DIR}/immediate_${EXPERIMENT_NAME}_popsize${POP_SIZE}_w_sweet.csv
