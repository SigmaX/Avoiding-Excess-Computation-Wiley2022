#!/usr/bin/env bash
#
# This drives the experiments.  The PDFs and CSV output will be in this
# directory after a run.  All output per iteration will be saved and will have
# the iteration number prepended to the filename.
#

PYTHONPATH=.:$PYTHONPATH

for run in {0..9}
  do
    # Do initial training of 2000 episodes
    export GREMLIN_QLEARNER_CARTPOLE_MODEL_FPATH="./output/run${run}_gremlin-enhanced-iter0_q-table.pkl"
    python train.py --model-fpath $GREMLIN_QLEARNER_CARTPOLE_MODEL_FPATH --num-episodes 5000

    # 2000 * 17 = 34000 episodes
    for iteration in {1..17}
    do
      echo "starting $iteration";

      # gremlin will generate pop.csv used by analyzer; inds.csv contains
      # all offspring, which is used by the analyzer
      gremlin.py -d config.yaml

      python analyzer.py --figures -n 100 --outfile stats.json inds.csv

      # Now resume training but using the distributions found in stats.json
      python train.py --stats stats.json --model-fpath $GREMLIN_QLEARNER_CARTPOLE_MODEL_FPATH --num-episodes 2000
      export GREMLIN_QLEARNER_CARTPOLE_MODEL_FPATH="./output/run${run}_gremlin-enhanced-iter${iteration}_q-table.pkl"

      # Now rename output from this iteration so it won't be overwritten
      for f in *.json *.pdf *.csv; do
        mv $f "./output/${run}_${iteration}_${f}"
      done

    done
done
