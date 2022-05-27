#!/usr/bin/env python3
"""
    DistributedIndividual subclass to allow for using getting birth ID support,
    and setting up for tracking eval start and stop times.
"""
import itertools
from time import time

# We are going to use Individual, instead, so that exceptions are not masked
from leap_ec.distrib.individual import DistributedIndividual
# from leap_ec.individual import Individual


class BalanceIndividual(DistributedIndividual):
    """ We inherit from DistributedIndividual so that exceptions are caught and
        processed such that fitness will be NaNs.  We are also setting up
        for additional bookkeeping for eval start and stop times that are
        managed by that class, as well as tracking unique birth IDs when we
        start using leap_ec.distrib EAs.
    """
    # Tracks unique birth ID for each newly created individual
    birth_id = itertools.count()

    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)

        # self.birth_id = next(BalanceIndividual.birth_id)
        self.start_eval_time = None
        self.stop_eval_time = None

    def __str__(self):
        phenome = self.decode()
        return f'{self.birth_id}, {phenome}, {self.fitness}'

    def evaluate(self):
        """ Overriding to capture how long evals take

            Don't have to do this; it's just here to show how it could be done.
        """
        self.start_eval_time = time()
        fitness = super().evaluate()
        self.stop_eval_time = time()

        return fitness
