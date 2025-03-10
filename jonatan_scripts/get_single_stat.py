import os
import re

import numpy as np
from variants import VARIANTS


def get_stat(input_file_path, stat_pattern, return_all=False):
    """
    Get statistics from a file matching the given pattern.

    Args:
        input_file_path (str): Path to the input file
        stat_pattern (str): Regular expression pattern to match
        return_all (bool): If True, returns all matches as a list; if False, returns first match

    Returns:
        If return_all=False: float or None - the first matching value or None if no matches
        If return_all=True: list - list of all matching values (empty if no matches)
    """
    res = []
    pattern = re.compile(stat_pattern)

    with open(input_file_path) as file:
        for line in file:
            if pattern.match(line):
                parts = line.split()
                if len(parts) > 1 and parts[1] != "nan":
                    if not return_all:
                        return float(parts[1])
                    res.append(float(parts[1]))

    return res if return_all else None


def get_single_stat_all_runs(benchmark, stat_name, return_all=False):
    # folder structure: jonatan_runs/<variant>/<benchmark>/<timestamp>/
    # get single stat from the file from all runs for a specific benchmark, and print the result per config
    # save the result in a list, per variant (dict)
    print("is it multi value?", return_all)
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
                stat_value = get_stat(stats_file, stat_name, return_all)
                if stat_value is not None:
                    res[variant].append(stat_value)
    #     if res[variant]:
    #         avg = np.mean(res[variant])
    #         std = np.std(res[variant])
    #         stats[variant] = (avg, std)
    #     else:
    #         stats[variant] = (None, None)

    # return stats, len(res[next(iter(res))])  # number of runs
    return res
