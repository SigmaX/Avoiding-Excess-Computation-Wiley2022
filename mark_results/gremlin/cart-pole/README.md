# Directory Contents

## Python

* `analytics.py` -- support module for jupyter notebooks for reading files
* `analyzer.py` -- analyzes Gremlin output to create JSON of distributions 
  for the four variables that is then used to resume training in regions of 
  interest for the OpenAI Gym cart-pole problem
* `individual.py` -- how we represent a single posed solution for the cart-pole 
  problem
* `probe.py` -- LEAP output probes
* `q_leaner_cartpoleev1.py` -- Q-learner for OpenAI Gym cart-pole problem
* `repesentation.py` -- how we represent solutions for cart-pole problem
* `sweet_gremlin.py` -- Gremlin hacked to implement SWEET algorithm
* `train.py` -- trains a Q-learner for the cart-pole problem

## Jupyter Notebooks

* `gremlin_analysis.ipynb` -- Gremlin run analysis and visualization
* `learner_analysis.ipynb` -- Q-learner performance analysis and visualization

## Shell

* `main.sh` -- run Gremlin w/o SWEET
* `sweet_main.sh` -- run Gremlin w/ SWEET

## YAML

* `config.yaml` -- Gremlin config file w/o SWEET
* `sweet_config.yaml` -- Gremlin config file w/ SWEET


# Set up for development

1. Checkout [Gremlin](https://github.com/markcoletti/gremlin/tree/develop); 
   be sure to be on the `develop` branch.
2. Activate your `conda` or `venv` environment.
3. `pip install .` from Gremlin top-level directory to install it into your 
   environment.
4. `gremlin.py --help` to test if it was successfully installed.

# Typical run cycle

This describes the steps to run Gremlin to optimize training for the OpenAI 
Gym CartPole problem.

1. Setup `config.yaml` to desired run-time parameters
2. While not done
   1. `train.py` to train a CartPole model
   2. `gremlin.py -d config.yaml`
      > This will generate a `pop.csv` containing the population snapshots for 
            each generation
   3. (TBD -- this is the step where we leverage the information from `pop.csv` 
      to improve training data)
