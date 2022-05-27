import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Asynchronous Heatmap Plotter")
    parser.add_argument("--results_file", "-f", required=True, type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.results_file)

    overall_stop_time = max(df["stop_eval_time"])
    overall_start_time = min(df["start_eval_time"])
    total_elapsed = overall_stop_time-overall_start_time
    max_fitness = max(df["fitness"])
    print("Maximum fitness:", max_fitness)
    print("Total Elapsed:", total_elapsed, np.ceil(total_elapsed))

    workers = sorted(np.unique(df["pid"]))
    A = np.zeros((len(workers),int(np.ceil(total_elapsed))))
    A.fill(np.nan)
    for i in range(len(df["hostname"])):
        w = workers.index(df["pid"][i])
        start_time = int(df["start_eval_time"][i]-overall_start_time)
        stop_time = int(df["stop_eval_time"][i]-overall_start_time)
        for j in range(start_time, stop_time+1):
            A[w][j] = df["fitness"][i]

    plt.pcolor(A)
    plt.xlim([0,1000])
    plt.colorbar(label="Fitness")
    plt.ylabel("Processor")
    plt.xlabel("Time (in seconds)")

    plt.show() 
