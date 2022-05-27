#!/usr/bin/env python3
"""
gremlin.py

THIS IS A CLONE OF THE ORIGINAL GREMLIN TO ALLOW FOR ADDING SWEET (Selection
While Evaluating).  The current version of Gremlin doesn't easily allow for the
highly tailored bookkeeping and operators needed to support SWEET.  So, here
we are.

usage: gremlin.py [-h] [-d] config_files [config_files ...]

Gremlin finds features sets where a given machine learning model performs
poorly.

positional arguments:
  config_files  path to configuration file(s) which Gremlin uses to set up the
                problem and algorithm

optional arguments:
  -h, --help    show this help message and exit
  -d, --debug   enable debugging output
"""
import sys
import multiprocessing
import toolz
from toolz import curry

# So we can pick up local modules defined in the YAML config file.
sys.path.append('.')

import argparse
import logging
import importlib
from typing import List, Iterator
import random

from omegaconf import OmegaConf

from rich.logging import RichHandler

# Create unique logger for this namespace
rich_handler = RichHandler(rich_tracebacks=True,
                           markup=True)
logging.basicConfig(level='INFO', format='%(message)s',
                    datefmt="[%Y/%m/%d %H:%M:%S]",
                    handlers=[rich_handler])
logger = logging.getLogger(__name__)

from rich import print
from rich import pretty
from rich.pretty import pprint

pretty.install()

from rich.traceback import install

install()

from distributed import Client, LocalCluster

from leap_ec.algorithm import generational_ea
from leap_ec.probe import AttributesCSVProbe
from leap_ec.global_vars import context
from leap_ec import ops, util
from leap_ec.ops import listiter_op

from leap_ec.int_rep.ops import mutate_randint, mutate_binomial
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.distrib import DistributedIndividual
# from leap_ec.distrib import asynchronous
from leap_ec.distrib.probe import log_worker_location, log_pop
from leap_ec.distrib.asynchronous import tournament_insert_into_pop, greedy_insert_into_pop, eval_population
from leap_ec.distrib.evaluate import evaluate, is_viable
from leap_ec.global_vars import context

from toolz import pipe


@toolz.curry
@listiter_op
def random_multiselection(population: List,
                          extra_population: List) -> Iterator:
    """ return a uniformly randomly selected individual from more than one
    population

    Treat the two populations as a single population from which to sample.

    Used to select from evaluated parents and currently evaluating individuals.

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
                                      inserter=greedy_insert_into_pop,
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
    evaluating_pop = {x.uuid: x for x in initial_population}

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
            logging.warning(
                f'UUID not found for newly evaluated individual {evaluated.uuid}')

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
                                   random_multiselection(extra_population=list(
                                       evaluating_pop.values())),
                                   which_pop(pop=pop,
                                             evaluating_pop=evaluating_pop),
                                   *offspring_pipeline)

            logging.debug('created offspring: ')
            [logging.debug('%s', str(o.genome)) for o in offspring]

            # New offspring is about to begin evaluation, so add 'em to the
            # evaluating set.
            evaluating_pop.update({x.uuid: x for x in offspring})

            # Now asynchronously submit to dask
            for child in offspring:
                future = client.submit(evaluate(context=context), child,
                                       pure=False)
                as_completed_iter.add(future)

            birth_counter(len(offspring))

    return pop


def read_config_files(config_files):
    """  Read one or more YAML files containing configuration options.

    The notion is that you can have a set of YAML files for controlling the
    configuration, such as having a set of default global settings that are
    overridden or extended by subsequent configuration files.

    E.g.,

    gremlin.py general.yaml this_model.yaml

    :param config_files: command line arguments
    :return: config object of current config
    """
    serial_configs = [OmegaConf.load(x) for x in config_files]
    config = OmegaConf.merge(*serial_configs)

    return config


def parse_config(config):
    """ Extract the population size, maximum generations to run, the Problem
    subclass, and the Representation subclass from the given `config` object.

    :param config: OmegaConf configurations read from YAML files
    :returns: Problem objects, Representation objects, LEAP pipeline operators
    """
    # The problem and representations will be something like
    # problem.MNIST_Problem, in the config and we just want to import
    # problem. So we snip out "problem" from that string and import that.
    globals()['problem'] = importlib.import_module(config.problem.split('.')[0])
    globals()['representation'] = importlib.import_module(
        config.representation.split('.')[0])

    if 'imports' in config:
        for extra_module in config.imports:
            globals()[extra_module] = importlib.import_module(extra_module)

    # Now instantiate the problem and representation objects, including any
    # ctor arguments.
    problem_obj = eval(config.problem)
    representation_obj = eval(config.representation)

    # Eval each pipeline function to build the LEAP operator pipeline
    pipeline = [eval(x) for x in config.pipeline]

    return problem_obj, representation_obj, pipeline


def run_generational_ea(pop_size, max_generations, problem, representation,
                        pipeline,
                        pop_file, k_elites=1):
    """ evolve solutions that show worse performing feature sets using a
    by-generation evolutionary algorithm (as opposed to an asynchronous,
    steady state evolutionary algorithm)

    :param pop_size: population size
    :param max_generations: how many generations to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline: LEAP operator pipeline to be used in EA
    :param pop_file: where to write the population CSV file
    :param k_elites: keep k elites
    :returns: None
    """
    with open(pop_file, 'w') as pop_csv_file:
        # Taken from leap_ec.algorithm.generational_ea and modified pipeline
        # slightly to allow for printing population *after* elites are included
        # in survival selection to get accurate snapshot of parents for next
        # generation.

        # If birth_id is an attribute, print that column, too.
        attributes = ('birth_id',) if hasattr(representation.individual_cls,
                                              'birth_id') else []

        pop_probe = AttributesCSVProbe(stream=pop_csv_file,
                                       attributes=attributes,
                                       do_genome=True,
                                       do_fitness=True)

        # Initialize a population of pop_size individuals of the same type as
        # individual_cls
        parents = representation.create_population(pop_size, problem=problem)

        # Set up a generation counter that records the current generation to
        # context
        generation_counter = util.inc_generation(
            start_generation=0, context=context)

        # Evaluate initial population
        parents = representation.individual_cls.evaluate_population(parents)

        print('Best so far:')
        print('Generation, str(individual), fitness')
        bsf = max(parents)
        print(0, bsf)

        pop_probe(parents)  # print out the parents and increment gen counter
        generation_counter()

        while (generation_counter.generation() < max_generations):
            # Execute the operators to create a new offspring population
            offspring = pipe(parents, *pipeline,
                             ops.elitist_survival(parents=parents,
                                                  k=k_elites),
                             pop_probe
                             )

            if max(offspring) > bsf:  # Update the best-so-far individual
                bsf = max(offspring)

            parents = offspring  # Replace parents with offspring
            generation_counter()  # Increment to the next generation

            # Output the best-so-far individual for each generation
            print(generation_counter.generation(), bsf)


def run_async_ea(pop_size, init_pop_size, max_births, problem, representation,
                 pipeline,
                 pop_file,
                 ind_file,
                 ind_file_probe,
                 scheduler_file=None):
    """ evolve solutions that show worse performing feature sets using an
    asynchronous steady state evolutionary algorithm (as opposed to a by-
    generation EA)

    :param pop_size: population size
    :param init_pop_size: the size of the initial random population, which
        can be different from the constantly updated population size that is
        dictated by `pop_size`; this is generally set to the number of
        available workers, but doesn't have to be
    :param max_births: how many births to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline: LEAP operator pipeline to be used to create a
        **single offspring**
    :param pop_file: where to write the CSV file of snapshot of population
        given every `pop_size` births
    :param ind_file: where to write the CSV file of each individual just as
        it is evaluated
    :param ind_file_probe: optional function (or functor) for printing out
        individuals to ind_file; if not specified, then
        `leap_ec.distrib.probe.log_worker_location` is used by default
    :param scheduler_file: optional dask scheduler file; will use cores on local
        host if none given
    :returns: None
    """
    if scheduler_file:
        logger.debug('Using cluster for dask')
    else:
        logger.debug('Using all localhost cores for dask')

    track_pop_stream = open(pop_file, 'w')
    track_pop_func = log_pop(pop_size, track_pop_stream)

    track_ind_func = None
    if ind_file is not None:
        if ind_file_probe is None:
            track_ind_stream = open(ind_file, 'w')
            track_ind_func = log_worker_location(track_ind_stream)
        else:
            track_ind_func = eval(ind_file_probe + '(open(ind_file,"w"))')

    if scheduler_file is None:
        logger.info('Using local cluster')
        cluster = LocalCluster(n_workers=multiprocessing.cpu_count(),
                               threads_per_worker=1,
                               processes=True,
                               silence_logs=logger.level)
        with Client(cluster) as client:
            final_pop = async_steady_state_eval_selection(client,
                                                  births=max_births,
                                                  init_pop_size=init_pop_size,
                                                  pop_size=pop_size,

                                                  representation=representation,

                                                  problem=problem,

                                                  offspring_pipeline=pipeline,

                                                  evaluated_probe=track_ind_func,
                                                  pop_probe=track_pop_func)

            print('Final pop:')
            print([str(x) for x in final_pop])
    else:
        logger.info('Using remote cluster')
        with Client(scheduler_file=scheduler_file,
                    processes=True,
                    silence_logs=logger.level) as client:
            final_pop = async_steady_state_eval_selection(client,
                                                  births=max_births,
                                                  init_pop_size=init_pop_size,
                                                  pop_size=pop_size,

                                                  representation=representation,

                                                  problem=problem,

                                                  offspring_pipeline=pipeline,

                                                  evaluated_probe=track_ind_func,
                                                  pop_probe=track_pop_func)

            print('Final pop:')
            print([str(x) for x in final_pop])


if __name__ == '__main__':
    logger.info('Gremlin started')

    parser = argparse.ArgumentParser(
        description=('Gremlin finds features sets where a given machine '
                     'learning model performs poorly.'))
    parser.add_argument('-d', '--debug',
                        default=False, action='store_true',
                        help=('enable debugging output'))
    parser.add_argument('config_files', type=str, nargs='+',
                        help=('path to configuration file(s) which Gremlin '
                              'uses to set up the problem and algorithm'))
    args = parser.parse_args()

    # set logger to debug if flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Logging set to DEBUG.')

    # combine configuration files into one dictionary
    config = read_config_files(args.config_files)
    logger.debug(f'Configuration: {config}')

    # Import the Problem and Representation classes specified in the
    # config file(s) as well as the LEAP pipeline of operators
    problem, representation, pipeline = parse_config(config)

    pop_size = int(config.pop_size)

    if config.algorithm == 'async':
        logger.debug('Using async EA')

        scheduler_file = None if 'scheduler_file' not in config['async'] else \
        config['async'].scheduler_file

        ind_file = None if 'ind_file' not in config['async'] else \
            config['async'].ind_file

        ind_file_probe = None if 'ind_file_probe' not in config['async'] else \
            config['async'].ind_file_probe

        run_async_ea(pop_size,
                     int(config['async'].init_pop_size),
                     int(config['async'].max_births),
                     problem, representation, pipeline,
                     config.pop_file,
                     ind_file,
                     ind_file_probe,
                     scheduler_file)
    elif config.algorithm == 'bygen':
        # default to by generation approach
        logger.debug('Using by-generation EA')

        # Then run leap_ec.generational_ea() with those classes while writing
        # the output to CSV and other, ancillary files.
        max_generations = int(config.bygen.max_generations)
        k_elites = int(config.bygen.k_elites) if 'k_elites' in config else 1

        run_generational_ea(pop_size, max_generations, problem, representation,
                            pipeline,
                            config.pop_file, k_elites)
    else:
        logger.critical(f'Algorithm type {config.algorithm} not supported')
        sys.exit(1)

    logger.info('Gremlin finished.')
