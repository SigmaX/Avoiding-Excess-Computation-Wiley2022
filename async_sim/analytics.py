#!/usr/bin/env python3
"""
    Consolidating common code here for notebook analytics/visualization.
"""
import pandas as pd
from pathlib import Path
import re

def consolidate_csv_as_df(glob_path, infer_run=False):
    """ Convenience for importing a collection of CSV files into a single
    pandas dataframe.

    >>> my_df = consolidate_csv_as_df(Path('.').glob('*csv'))

    :param glob_path: Path.glob() collection of CSV files to be consolidating
    into a single dataframe
    :param infer_run: if True, try to infer run number from CSV filename
    :return: pandas dataframe
    """
    data_frames = []
    for csv_file in glob_path:
        #print(f'reading {csv_file}')
        data_frames.append(pd.read_csv(str(csv_file)))

        if infer_run:
            matched_run_num = re.search('.*(\d).*', str(csv_file))
            if matched_run_num:
                run_num = matched_run_num.group(1)
                #print(f'Adding run number {run_num}')
                data_frames[-1]['run'] = int(run_num)

    return pd.concat(data_frames)


def idle_vs_work_times(df):
    """ Return a dataframe of the idle and work totals

    Infer work and idle times by serializing the time series data of
    start/stop pairs by unique worker id.  Interleaved will be the time
    spent doing evaluation and the time in between work, or idle time.

    :param df: dataframe that has start and stop times
    """
    df['worker_id'] = df['hostname'] + df['pid'].astype(str)

    # %%

    # Turn the NaNs to zeros so we can at least count them
    # df['fitness'] = df['fitness'].fillna(0)

    # %%

    # group by the worker id so we can extract the start/stop times for that worker
    by_worker = df.groupby('worker_id')

    # %%

    # Dictionary keyed by worker id that will contain a list of start/stop times for each of them
    worker_time_series = {}

    for name, group in by_worker:
        print(name)

        # We will serialize the start/stop times in 'times'; later, we'll add a sibling key to categorize each interval
        # and track their lengths.
        worker_time_series[name] = {'times'   : [],
                                    'run_type': group['run_type'].iloc[0]}

        for start, stop in zip(group['start_eval_time'],
                               group['stop_eval_time']):
            print(f'{start} {stop}')
            worker_time_series[name]['times'].append(start)
            worker_time_series[name]['times'].append(stop)

    # %%

    # Now lets convert that time series data to actual times series data format
    for worker_id in worker_time_series.keys():
        # First we convert to a Series so that when we next convert to datetime it will return that as a series
        # instead of a DatetimeIndex.  We don't want an index, we want a Series so that we have access to
        # rolling(), which does not exist for DatetimeIndex objects.
        worker_time_series[worker_id]['times'] = pd.Series(
            worker_time_series[worker_id]['times'])

        # Commented out the following because we just want to use subtract and that's not allowed on time series
        # worker_time_series[worker_id]['times'] = pd.to_datetime(worker_time_series[worker_id]['times'], unit='s')

        # Now compute the duration for each interval
        worker_time_series[worker_id]['durations'] = \
        worker_time_series[worker_id]['times'].rolling(2).apply(
            lambda x: x[1] - x[0], raw=True)

        # Extract the durations for just when we were busy
        worker_time_series[worker_id]['work_durations'] = \
        worker_time_series[worker_id]['durations'][1::2]
        worker_time_series[worker_id]['work_durations_total'] = \
        worker_time_series[worker_id]['work_durations'].sum()

        # And then extract the idle time durations
        worker_time_series[worker_id]['idle_durations'] = \
        worker_time_series[worker_id]['durations'][::2]
        worker_time_series[worker_id]['idle_durations_total'] = \
        worker_time_series[worker_id]['idle_durations'].sum()

    # %%

    return pd.DataFrame.from_dict(worker_time_series, orient='index')
