#!/usr/bin/env python3
"""
    This is a simple test problem whereby each individual has a single
    genotypic value that maps to an integer, i.  During evaluation a
    given individual will sleep for i seconds.  It's a maximization problem,
    so longer slumbering individuals have higher fitnesses.

    This test problem serves two purposes.  First, it does a better job of
    emulating real-world behavior in that evaluations can take a variable
    amount of time.  (The previous dask scaling problem was too simple to
    solve, and so all the evaluations pretty much ended simultaneously, which
    portrayed an un-realistic workload for the dask scheduler.)  Second, this
    gives us an opportunity to explore the dynamics of asynchronous steady-
    state EAs -- in this case, the population should theoretically be domin-
    ated by individuals with lower values since they'd finish evaluation sooner.
    However, note from Eric "Siggy" Scott of GMU suggest that in such scenarios
    that better individuals manage to eventually recover and dominate the
    population.  We'll see!
"""
from time import sleep

from leap_ec.problem import ScalarProblem


class SleeperProblem(ScalarProblem):
    """
    Will have individuals sleep for N seconds, where N is the phenome.
    """
    def __init__(self, verbose=True):
        """
        :param verbose: boolean for chatting behavior during eval
        """
        super().__init__(maximize=True)

        self.verbose = verbose

    def evaluate(self, phenome):
        """
        :param phenome: integer value, i; will cause this to sleep(i)
        :return: that integer value
        """
        # phenome[0] because we get a vector of just one integer
        sleep(phenome[0])

        return phenome[0]

    def __str__(self):
        return SleeperProblem.__name__
