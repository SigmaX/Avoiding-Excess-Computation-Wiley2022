#!/usr/bin/env python3
"""
    This defines a LEAP Problem subclass for the balancing pole problem.

    The BalanceProblem will decode an individual representing a parameter for the
    model, run the model, and return the accuracy.  This presumes that a
    balancing pole model has already been trained via `train.py`.
"""
from pathlib import Path
import sre_compile
from typing import Callable

from leap_ec.problem import ScalarProblem
from numpy import nan
import numpy as np
import gym
from gym.envs.classic_control.cartpole import CartPoleEnv
from q_learner_cartpolev1 import CartPoleV1_Q_Learner

from representation import BalancePhenotype


class BalanceProblem(ScalarProblem):
    """LEAP Problem subclass that encapsulates a model
    inference used to evaluate how well it can do the OpenAI Gym balancing
    pole problem
    """

    # Where the trained model is located
    data_path = Path("..") / "data"
    model_chk_file = data_path / "model.pt"  # FIXME this name is likely wrong

    def __init__(self):
        """
        We are _minimizing_ for accuracy; alternatively we could have
        maximized for loss. I.e., gremlin wants to find where the model
        performs poorly, not the best.
        """
        super().__init__(maximize=False)
        # TODO load model
        # TODO load test dataset??  May not be necessary for pole problem
        # TODO any other bookkeeping to set up for inference/predict
        # Set up the OpenAI Gym cart pole
        self._env: CartPoleEnv = gym.make("CartPole-v1")

    def evaluate(self, phenome):
        """
        Evaluate the phenome with the given model.

        :param phenome: is named tuple describing a balancing pole state
        :returns: score for model performance for this state
        """
        # TODO add code to run model for this state
        # TODO add code to calculate correctness
        # Override this bit
        NotImplementedError()
        return nan  # TODO replace with actual correctness metric


class QLearnerBalanceProblem(BalanceProblem):
    """LEAP Problem subclass that encapsulating Q Learner for
    OpenAI Gym balancing pole problem
    """

    def __init__(self, model_fpath: str):
        super().__init__()
        self.model_chk_file = Path(model_fpath)
        # TODO load model
        # TODO load test dataset??  May not be necessary for pole problem
        # TODO any other bookkeeping to set up for inference/predict

    def evaluate(self, phenome: BalancePhenotype) -> int:
        """
        Evaluate the phenome with the given model.

        :param phenome: is named tuple describing a balancing pole state
        :returns: score for model performance for this state
        """
        model = CartPoleV1_Q_Learner()
        model.load(self.model_chk_file)
        fitness = model.predict(list(phenome))
        return fitness
