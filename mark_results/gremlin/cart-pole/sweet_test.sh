#!/usr/bin/env bash
#
# Testbed for working out dask issues
#

PYTHONPATH=.:$PYTHONPATH

export GREMLIN_QLEARNER_CARTPOLE_MODEL_FPATH="./output/run0_gremlin-enhanced-iter0_q-table.pkl"
python train.py --model-fpath $GREMLIN_QLEARNER_CARTPOLE_MODEL_FPATH --num-episodes 2000

#dask-scheduler --scheduler-file scheduler.json &

#for i in {0..4}; do
#  dask-worker --preload individual.py --nthreads 1 --nprocs 4 --scheduler-file scheduler.json &
#done

./sweet_gremlin.py -d sweet_config.yaml

#pkill -HUP -f dask-worker
#pkill -HUP -f dask-scheduler
