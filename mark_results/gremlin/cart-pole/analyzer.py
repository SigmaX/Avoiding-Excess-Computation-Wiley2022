#!/usr/bin/env python3
""" Analyzes gremlin output to compute ranges for training.

    This is both an importable module *and* a standalone executable.  In the,
    latter it's intended as a debug tool to "see" what the trainer will see
    with regards to valid exploration ranges for each gene.

usage: analyzer.py [-h] [--outfile OUTFILE] [-d] [-f] [-n N] input_file

Gremlin output analyzer

positional arguments:
  input_file            Gremlin CSV output file of each generation

optional arguments:
  -h, --help            show this help message and exit
  --outfile OUTFILE, -o OUTFILE
                        Where to write the regions of interest json file; if
                        left out, will default to writing to stdout
  -d, --drop-duplicates
                        Drop duplicates by birth_id from distribution
                        calculations
  -f, --figures         Optionally save figures showing the distributions for
                        all four genes
  -n N                  How many of the worst to consider for evaluation for
                        distribution
"""
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme() # set to default Seaborn theme.


def to_array(s):
    """ s is string rep of np array

        Used to convert the string of a numpy array for our genome into
        four distinct values that can be later converted into dataframe columns.
    """
    as_array = np.fromstring(s[1:-1], dtype=float, sep=' ')
    return as_array[0], as_array[1], as_array[2], as_array[3]


def write_regions(gremlin_file, n, drop_duplicates, save_figures, outstream):
    """ This will scan the gremlin CSV and write out the regions of weakness
    to be further explored with the RL.

    TODO `write_regions` is not a good name for this function

    :param gremlin_file: CSV file from a gremlin run, likely pop.csv
    :param n: how many of the worst to consider for defining region
    :param drop_duplicates: if true, then we drop duplicates by birth_id
    :param save_figures: if true, write out plots of distributions for all
        four genes
    :param outstream: stream to write JSON for describing the mean and std for
        each variable
    :return: None
    """
    def write_json(worst_n):
        """ Write out worst N stats to JSON to given stream """
        results = {'position'           : {},
                   'velocity'           : {},
                   'angle'              : {},
                   'rotational_velocity': {}}
        results['position']['mean'] = worst_n['position'].mean()
        results['position']['median'] = worst_n['position'].median()
        results['position']['std'] = worst_n['position'].std()
        results['velocity']['mean'] = worst_n['velocity'].mean()
        results['velocity']['median'] = worst_n['velocity'].median()
        results['velocity']['std'] = worst_n['velocity'].std()
        results['angle']['mean'] = worst_n['angle'].mean()
        results['angle']['median'] = worst_n['angle'].median()
        results['angle']['std'] = worst_n['angle'].std()
        results['rotational_velocity']['mean'] = worst_n[
            'rotational_velocity'].mean()
        results['rotational_velocity']['median'] = worst_n[
            'rotational_velocity'].median()
        results['rotational_velocity']['std'] = worst_n[
            'rotational_velocity'].std()

        outstream.write(json.dumps(results, sort_keys=True, indent=4))

    def write_figs(worst_n):
        """ Write out distributions for each gene to PDFs """
        sns.histplot(data=worst_n, x='position', binrange=(-2.5,2.5)).set(
            xlabel='Cart Position')
        plt.savefig('pos_dist.pdf')
        plt.clf()

        sns.histplot(data=worst_n, x='velocity', binrange=(-0.05,0.05)).set(
            xlabel='Velocity')
        plt.savefig('vel_dist.pdf')
        plt.clf()

        sns.histplot(data=worst_n, x='angle', binrange=(-0.21,0.21)).set(
            xlabel='Pole Angle')
        plt.savefig('angle_dist.pdf')
        plt.clf()

        sns.histplot(data=worst_n, x='rotational_velocity', binrange=(-0.05,0.05)).set(
            xlabel='Pole Rotational Velocity')
        plt.savefig('rot_vel_dist.pdf')
        plt.clf()


    gremlin_output = pd.read_csv(gremlin_file)

    # The better individuals will show up more than once, so
    # we want to cull the duplicates.
    if drop_duplicates:
        gremlin_output = gremlin_output.drop_duplicates('birth_id')
        print(f'Duplicates were {(1 - (len(gremlin_output) / 5000)) * 100:0.4}% of data',
              file=sys.stderr)
    else:
        print('Not dropping duplicates', file=sys.stderr)

    # Now sort by fitness so we can snip out the "worst" N entries
    gremlin_output = gremlin_output.sort_values(by=['fitness'])

    # So the fitnesses are being minimized as they should. Now to look at the
    # distribution of the genes. But first we have to break out the genes
    # into separate columns.

    # Not needed since the genome has already been broken out
    # scratch_df = pd.DataFrame(gremlin_output['genome'].apply(to_array))
    # gremlin_output[['position', 'velocity', 'angle',
    #                 'rotational_velocity']] = pd.DataFrame(
    #     scratch_df.genome.tolist(), index=gremlin_output.index)

    worst_n = gremlin_output.head(n)
    write_json(worst_n)

    if save_figures:
        write_figs(worst_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gremlin output analyzer')
    parser.add_argument('--outfile', '-o', required=False,
                        help='Where to write the regions of interest json file; if left out, will default to writing to stdout')
    parser.add_argument('-d', '--drop-duplicates', action='store_true',
                        help='Drop duplicates by birth_id from distribution calculations')
    parser.add_argument('-f', '--figures', action='store_true',
                        help='Optionally save figures showing the distributions for all four genes')
    parser.add_argument('-n', default=5, type=int, help='How many of the worst to consider for evaluation for distribution')
    parser.add_argument('input_file',
                        help='Gremlin CSV output file of each generation')

    args = parser.parse_args()

    if args.outfile is None:
        write_regions(args.input_file, args.n, args.drop_duplicates, args.figures, sys.stdout)
    else:
        with open(args.outfile, 'w') as outstream:
            write_regions(args.input_file, args.n, args.drop_duplicates, args.figures, outstream)
