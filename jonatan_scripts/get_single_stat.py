import os

import numpy as np
from variants import VARIANTS


def get_single_stat(input_file_path, stat_name):
    with open(input_file_path) as file:
        for line in file:
            if stat_name in line:
                # Assuming the format is "stat_name value # extra_info"
                parts = line.split()
                if len(parts) > 1 and parts[1] != "nan":
                    return float(parts[1])
    return None


def get_single_stat_all_runs(benchmark, stat_name):
    # folder structure: jonatan_runs/<variant>/<benchmark>/<timestamp>/
    # get single stat from the file from all runs for a specific benchmark, and print the result per config
    # save the result in a list, per variant (dict)
    res = {}
    stats = {}
    for variant, _1, _2 in VARIANTS:
        res[variant] = []
        variant_path = f"jonatan_runs/{variant}/{benchmark}"
        if not os.path.exists(variant_path):
            continue  # Skip if the path does not exist
        for run in os.listdir(variant_path):
            stats_file = f"{variant_path}/{run}/stats.txt"
            if os.path.exists(stats_file):
                stat_value = get_single_stat(stats_file, stat_name)
                if stat_value is not None:
                    res[variant].append(stat_value)
        if res[variant]:
            avg = np.mean(res[variant])
            std = np.std(res[variant])
            stats[variant] = (avg, std)
        else:
            stats[variant] = (None, None)

    return stats, len(res[next(iter(res))])  # number of runs
