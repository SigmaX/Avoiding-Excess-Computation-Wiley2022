#!/usr/bin/env python3
"""
    Test bed to work out approach for including evaluating individuals
    when selecting for parents.  Essentially union evaluating individuals and
    those in the population and randomly select from that.
"""
import logging
logging.basicConfig(level=logging.INFO)

import random
from typing import List, Iterator

import toolz

from distributed import Client, as_completed

from leap_ec import util, ops
from leap_ec.ops import listiter_op
from leap_ec.distrib.individual import DistributedIndividual
from leap_ec.representation import Representation
from leap_ec.distrib.asynchronous import tournament_insert_into_pop, eval_population
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.distrib.evaluate import evaluate, is_viable
from leap_ec.int_rep.ops import mutate_randint
from leap_ec.global_vars import context

from sleep_problem import SleeperProblem


@toolz.curry
@listiter_op
def random_multiselection(population: List,
                          extra_population: List) -> Iterator:
    """ return a uniformly randomly selected individual from more than one
    population

    Treat the two populations as a single population from which to sample.

    :param population: from which to select
    :param extra_population: another from which to select
    :return: a uniformly selected individual
    """
    while True:
        yield random.choice(population + extra_population)

@toolz.curry
def which_pop(next_individual, pop, evaluating_pop):
    """ Reports with population the given individual came from

    :param next_individual: to come down the pipeline
    :param pop: is the current parent population
    :param evaluating_pop: is the dictionary of individuals that are currently
        being evaluating keyed by UUID
    """
    while True:
        individual = next(next_individual)

        pop_by_keys = {x.uuid : x for x in pop}

        if individual.uuid in pop_by_keys and individual.uuid in evaluating_pop:
            logging.debug('IN both!!')
        elif individual.uuid in pop_by_keys:
            logging.debug('IN pop')
        elif individual.uuid in evaluating_pop:
            logging.debug('IN evaluating pop')
        else:
            logging.debug('IN neither')

        yield individual

def async_steady_state_eval_selection(client, births, init_pop_size, pop_size,
                                      representation,
                                      problem, offspring_pipeline,
                                      inserter=tournament_insert_into_pop,
                                      count_nonviable=False,
                                      evaluated_probe=None,
                                      pop_probe=None,
                                      context=context):
    """ Implements an asynchronous steady-state EA

    NOTE: this was ganked from leap_ec.distrib.asynchronous

    :param client: Dask client that should already be set-up
    :param births: how many births are we allowing?
    :param init_pop_size: size of initial population sent directly to workers
           at start
    :param pop_size: how large should the population be?
    :param representation: of the individuals
    :param problem: to be solved
    :param offspring_pipeline: for creating new offspring from the pop
    :param inserter: function with signature (new_individual, pop, popsize)
           used to insert newly evaluated individuals into the population;
           defaults to tournament_insert_into_pop()
    :param count_nonviable: True if we want to count non-viable individuals
           towards the birth budget
    :param evaluated_probe: is a function taking an individual that is given
           the next evaluated individual; can be used to print newly evaluated
           individuals
    :param pop_probe: is an optional function that writes a snapshot of the
           population to a CSV formatted stream ever N births
    :return: the population containing the final individuals
    """
    initial_population = representation.create_population(init_pop_size,
                                                          problem=problem)

    # Remember the individuals we're about to fan out to begin evaluation; here
    # we create a dictionary keyed by the UUID associated with each individual
    evaluating_pop = {x.uuid : x for x in initial_population}

    # fan out the entire initial population to dask workers
    as_completed_iter = eval_population(initial_population, client=client,
                                        context=context)

    # This is where we'll be putting evaluated individuals
    pop = []

    # Bookkeeping for tracking the number of births
    birth_counter = util.inc_births(context, start=len(initial_population))

    for i, evaluated_future in enumerate(as_completed_iter):

        evaluated = evaluated_future.result()

        # This guy is finished evaluation, so remove him from the evaluated
        # population.
        if evaluated.uuid in evaluating_pop:
            del evaluating_pop[evaluated.uuid]
        else:
            logging.warning(f'UUID not found for newly evaluated individual {evaluated.uuid}')

        if evaluated_probe is not None:
            # Give a chance to do something extra with the newly evaluated
            # individual, which is *usually* a call to
            # probe.log_worker_location, but can be any function that
            # accepts an individual as an argument
            evaluated_probe(evaluated)

        logging.debug('%d evaluated: %s %s', i, str(evaluated.genome),
                     str(evaluated.fitness))

        if not count_nonviable and not is_viable(evaluated):
            # If we don't want non-viable individuals to count towards the
            # birth budget, then we need to decrement the birth count that was
            # incremented when it was created for this individual since it
            # was broken in some way.
            birth_counter()

        inserter(evaluated, pop, pop_size)

        if pop_probe is not None:
            pop_probe(pop)

        if birth_counter.births() < births:
            # Only create offspring if we have the budget for one
            offspring = toolz.pipe(pop,
                                   # We select from the union of the population
                                   # AND the currently evaluating individuals.
                                   random_multiselection(extra_population=list(evaluating_pop.values())),
                                   which_pop(pop=pop, evaluating_pop=evaluating_pop),
                                   *offspring_pipeline)

            logging.debug('created offspring: ')
            [logging.debug('%s', str(o.genome)) for o in offspring]

            # New offspring is about to begin evaluation, so add 'em to the
            # evaluating set.
            evaluating_pop.update({x.uuid : x for x in offspring})

            # Now asynchronously submit to dask
            for child in offspring:
                future = client.submit(evaluate(context=context), child,
                                       pure=False)
                as_completed_iter.add(future)

            birth_counter(len(offspring))

    return pop


if __name__ == '__main__':
    bounds = [(1, 5)]
    representation = Representation(initialize=create_int_vector(bounds),
                                    individual_cls=DistributedIndividual)

    with Client() as client:
        final_pop = async_steady_state_eval_selection(client, births=10,
                                                      init_pop_size=4,
                                                      pop_size=4,
                                                      representation=representation,
                                                      problem=SleeperProblem(),
                                                      offspring_pipeline=[
                                                          # NOTE NO RANDOM SELECTION HERE
                                                          # IT HAS BEEN MOVED INSIDE
                                                          # OF THIS FUNCTION
                                                          ops.clone,
                                                          mutate_randint(
                                                              bounds=bounds,
                                                              expected_num_mutations=1),
                                                          ops.pool(size=1)])

    print(final_pop)
