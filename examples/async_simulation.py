"""Run a discrete-event simulation of an asynchronous evolutionary
algorithm with different fitness landscapes and evaluation-time
functions.

Usage:
  async_simulation.py [options] [--immediate | --until-all | --extra]
  async_simulation.py -h | --help

Options:
  -h --help                     Show this screen.
  -g --births=<n>               Number of evolutionary steps to run for. [default: 2000]
  --pop-size=<n>                Number of individuals in each subpopulation. [default: 10]
  --no-viz                      Disable real-time visualization.
  --jobs=<n>                    Number of independent runs to perform. [default: 50]
  --immediate                   Use the "immediate" initialization strategy: all initial individuals are sent to the cluster, and we being steady-state evolution as soon as there is *one* individual in the population.
  --until-all                   Use the "until all evaluating" initialization strategy: all initial individuals are sent to the cluster, and we wait until all of them have been evaluated or are currently being evaluated before we begin steady-state evolution.
  --extra                       Use the "exra" initialization strategy: all initial individuals are sent to the cluster, plus (num_processors - 1) "extra" initial individuals, and we wait until all of the initial (non-exra) individuals have evaluated before we begin steady-state evolution.
  --num-processors=<n>          Number of processors in our simulated cluster; set this to 'pop-size' to have it automatically match the population size. [default: pop-size]
  --sweet                       Use the "selection while evaluating" strategy.
  --fitness-function=<name>     Name of the fitness function to use. [default: exponential-growth]
  --evaltime-function=<name>    Name of the evaluation-time function to use. [default: exponential-growth]
  --dimensions=<int>            Number of dimensions for the fitness and eval-time functions. [default: 10]
"""
import inspect
import os
import sys

from matplotlib import pyplot as plt
import numpy as np

from docopt import docopt
from leap_ec import ops, probe, Individual
from leap_ec import Representation
from leap_ec.algorithm import generational_ea
from leap_ec.problem import ConstantProblem, FunctionProblem
from leap_ec.real_rep import problems as real_prob
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian

from async_sim import components as co


##############################
# Entry point
##############################
if __name__ == '__main__':
    ##############################
    # Parameters
    ##############################
    # CLI parameters
    arguments = docopt(__doc__)
    births = int(arguments['--births'])
    viz = not bool(arguments['--no-viz'])
    jobs = int(arguments['--jobs'])
    pop_size = int(arguments['--pop-size'])
    sweet = bool(arguments['--sweet'])
    problem_name = arguments['--fitness-function']
    eval_time_name = arguments['--evaltime-function']
    dimensions = int(arguments['--dimensions'])

    if arguments['--num-processors'] == 'pop-size':
        num_processors = pop_size
    else:
        num_processors = int(arguments['--pop-size'])

    if bool(arguments['--immediate']):
        init_strategy = 'immediate'
    elif bool(arguments['--until-all']):
        init_strategy = 'until_all_evaluating'
    elif bool(arguments['--extra']):
        init_strategy = 'extra'
    else:
        # Default value if no option given
        init_strategy = 'extra'

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(co.test_env_var, False) == 'True':
        jobs = 2
        births = 2

    # Fixed parameters
    modulo = 10


    ##############################
    # Setup
    ##############################
    #experiment_note = f"\"{problem_name} fitness\n{eval_time_name} eval-time (no crossover)\""
    experiment_note = init_strategy
    problem = co.get_standard_function(problem_name, dimensions=dimensions)
    eval_time_prob = co.get_standard_function(eval_time_name, dimensions=dimensions)
    eval_time_f = lambda x: eval_time_prob.evaluate(Individual(x))
    
    if viz:
        plt.figure(figsize=(20, 8))


    ##############################
    # For each job
    ##############################
    with open('birth_times.csv', 'w') as births_file:
        for job_id in range(jobs):

            ##############################
            # Setup Metrics and Simulation
            ##############################
            if viz:  # Set up the top row of visuals
                pad_val = 0  # Value to fix higher-dimensional values at when projecting landscapes into 2-D visuals
                pad_vec = np.array([pad_val]*(dimensions - 2))
                plt.subplot(231, projection='3d')
                real_prob.plot_2d_problem(problem, xlim=problem.bounds, ylim=problem.bounds, pad=pad_vec, title="Fitness Landscape", ax=plt.gca())

                plt.subplot(232, projection='3d')
                real_prob.plot_2d_function(eval_time_f, xlim=problem.bounds, ylim=problem.bounds, pad=pad_vec, title="Eval-Time Landscape", ax=plt.gca())

                plt.subplot(233)  # Put the Gantt plot in the upper-right
                p = co.GanttPlotProbe(ax=plt.gca(), max_bars=100, modulo=modulo)

                gui_steadystate_probes = [ p ]
            else:
                gui_steadystate_probes = []
            
            # Set up the cluster simulation.
            # This mimics an asynchronous evaluation engine, which may return individuals in an order different than they were submitted.
            eval_cluster = co.AsyncClusterSimulation(
                                            num_processors=num_processors,
                                            eval_time_function=eval_time_f,
                                            # Individual-level probes (these run just on the newly evaluated individual)
                                            probes=[
                                                probe.AttributesCSVProbe(attributes=['birth', 'start_time', 'end_time', 'eval_time'],
                                                                         notes={ 'job': job_id }, header=(job_id==0), stream=births_file)
                                            ] + gui_steadystate_probes)

            # Set up the second row of visuals
            if viz:
                plt.subplot(234)
                p1 = probe.CartesianPhenotypePlotProbe(xlim=problem.bounds, ylim=problem.bounds,
                                                contours=problem, pad=[pad_val]*(dimensions - 2), ax=plt.gca(), modulo=modulo)
                
                plt.subplot(235)
                p2 = probe.FitnessPlotProbe(ax=plt.gca(), modulo=modulo,
                                            title="Best-of-step Fitness (by step).")
                
                plt.subplot(236)
                p3 = probe.FitnessPlotProbe(ax=plt.gca(), modulo=modulo,
                                            title="Best-of-step Fitness (by time).",
                                            x_axis_value=lambda: eval_cluster.time)

                # Leave the dashboard in its own window
                p4 = co.AsyncClusterDashboardProbe(cluster_sim=eval_cluster, modulo=modulo)

                gui_pop_probes = [ p1, p2, p3, p4 ]
            else:
                gui_pop_probes = []


            ##############################
            # Evolve
            ##############################

            # Defining representation up front, so we can use it a couple different places
            representation=Representation(
                            # Initialize a population of integer-vector genomes
                            initialize=create_real_vector(bounds=[problem.bounds] * dimensions)
                        )

            # Selection strategy
            if sweet:
                selection = co.select_with_processing(ops.random_selection, eval_cluster)
            else:
                selection = ops.random_selection
            
            ea = generational_ea(max_generations=births,pop_size=pop_size,
                                    
                                    # We use an asynchronous scheme to evaluate the initial population
                                    init_evaluate=co.async_init_evaluate(
                                        cluster_sim=eval_cluster,
                                        strategy=init_strategy,
                                        create_individual=lambda: representation.create_individual(problem)),
                                        
                                    problem=problem,  # Fitness function

                                    # Representation
                                    representation=representation,

                                    # Operator pipeline
                                    pipeline=[
                                        co.steady_state_step(
                                            reproduction_pipeline=[
                                                selection,
                                                ops.clone,
                                                #ops.uniform_crossover(p_swap=0.2),
                                                mutate_gaussian(std=1.5, hard_bounds=[problem.bounds]*dimensions,
                                                        expected_num_mutations=1)
                                            ],
                                            insert=co.competition_inserter(p_accept_even_if_worse=0.0, pop_size=pop_size,
                                                replacement_selector=ops.random_selection
                                            ),
                                            # This tells the steady-state algorithm to use asynchronous evaluation
                                            evaluation_op=co.async_evaluate(cluster_sim=eval_cluster)
                                        ),
                                        # Population-level probes (these run on all individuals in the population)
                                        probe.FitnessStatsCSVProbe(stream=sys.stdout, header=(job_id==0), modulo=modulo,
                                                comment=inspect.getsource(sys.modules[__name__]),  # Put the entire source code in the comments
                                                notes={'experiment': experiment_note, 'job': job_id},
                                                extra_metrics={
                                                    'time': lambda x: eval_cluster.time, 
                                                    'birth': lambda x: eval_cluster.birth,
                                                    'mean_eval_time': lambda x: np.mean([ind.eval_time for ind in x]),
                                                    'diversity': probe.pairwise_squared_distance_metric
                                                }
                                        ),
                                    ] + gui_pop_probes
                                )

            # Er, actually go!
            list(ea)
