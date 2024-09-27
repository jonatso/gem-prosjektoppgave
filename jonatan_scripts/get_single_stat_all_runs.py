import os
import re

from rodinia_list import BENCHMARKS


def get_single_stat(input_file_path, stat_name):
    # Define regex pattern for the desired format
    pattern = re.compile(rf"{stat_name}.*")

    # print to standard output
    with open(input_file_path) as file:
        for line in file:
            if pattern.match(line):
                match = re.search(r"([0-9]+\.[0-9]+|nan)", line)
                return float(match.group(1)) if match else None
    print(f"Stat {stat_name} not found in {input_file_path}")
    return None


def get_single_stat_all_runs(benchmark, stat_name):
    # folder structure: jonatan_runs/<variant>/<benchmark>/<timestamp>/
    # get single stat from the file from all runs for a specific benchmark, and print the result per config
    # save the result in a list, per variant (dict)
    res = {}
    for variant in os.listdir("jonatan_runs"):
        res[variant] = []
        for run in os.listdir(f"jonatan_runs/{variant}/{benchmark}"):
            stats_file = f"jonatan_runs/{variant}/{benchmark}/{run}/stats.txt"
            res[variant].append(get_single_stat(stats_file, stat_name))

    # get the average of the stats for each variant
    for variant in res:
        res[variant] = sum(res[variant]) / len(res[variant])
    return res


STAT_NAME = "system.cpu0.ipc"

if __name__ == "__main__":
    for benchmark in BENCHMARKS:
        stats = get_single_stat_all_runs(benchmark, STAT_NAME)
        print(f"Stats for {benchmark}:")
        print(stats)
