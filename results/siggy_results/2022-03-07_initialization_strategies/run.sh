#!/bin/bash

# CLI parameters
POP_SIZE=${1}
echo "Running with POP_SIZE=${POP_SIZE}"

# Fixed parameters
JOBS=100
BIRTHS=5000
OUTPUT_DIR=2022-03-07_initialization_strategies_results/
mkdir -p ${OUTPUT_DIR}

echo "Running immediate..."
python ../../../examples/async_simulation.py \
    --no-viz \
    --immediate \
    --pop-size=${POP_SIZE} \
    --jobs=${JOBS} \
    --births=${BIRTHS} \
    > ${OUTPUT_DIR}/immediate_popsize${POP_SIZE}.csv

echo "Running until-all..."
python ../../../examples/async_simulation.py \
    --no-viz \
    --until-all \
    --pop-size=${POP_SIZE} \
    --jobs=${JOBS} \
    --births=${BIRTHS} \
    > ${OUTPUT_DIR}/until_all_popsize${POP_SIZE}.csv

echo "Running extra..."
python ../../../examples/async_simulation.py \
    --no-viz \
    --extra \
    --pop-size=${POP_SIZE} \
    --jobs=${JOBS} \
    --births=${BIRTHS} \
    > ${OUTPUT_DIR}/extra_popsize${POP_SIZE}.csv