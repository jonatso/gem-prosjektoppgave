import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from dnnmark_list import BENCHMARKS
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
    for (variant, _) in VARIANTS:
        res[variant] = []
        for run in os.listdir(f"jonatan_runs/{variant}/{benchmark}"):
            stats_file = f"jonatan_runs/{variant}/{benchmark}/{run}/stats.txt"
            stat_value = get_single_stat(stats_file, stat_name)
            if stat_value is not None:
                res[variant].append(stat_value)
        if res[variant]:
            avg = np.mean(res[variant])
            std = np.std(res[variant])
            stats[variant] = (avg, std)
        else:
            stats[variant] = (None, None)

    return stats, len(res[VARIANTS[0][0]])  # number of runs

def plot_stats(stats, benchmark_name, stat_name, num_runs):
    plt.figure()
    variants = list(stats.keys())
    averages = [stats[variant][0] for variant in variants]
    std_devs = [stats[variant][1] for variant in variants]

    # Generate a list of colors for each variant
    colors = plt.cm.get_cmap('tab10', len(variants)).colors

    plt.bar(variants, averages, yerr=std_devs, capsize=5, color=colors, label=f"Average of {num_runs} runs")

    plt.xlabel("Variant")
    plt.ylabel(stat_name)
    plt.title(f"{stat_name} for {benchmark_name}")
    plt.legend()

    output_dir = f"jonatan_images/{stat_name}"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{benchmark_name}.png")
    plt.savefig(filename)
    plt.close()


STAT_NAME = "system.cpu1.shaderActiveTicks"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--stat",
        default=STAT_NAME,
        help="Stat to get",
    )
    args = parser.parse_args()

    for benchmark_name in BENCHMARKS.keys():
        stats, num_runs = get_single_stat_all_runs(
            benchmark_name, args.stat
        )
        print(f"Stats for {benchmark_name}:")
        print(stats)
        plot_stats(stats, benchmark_name, args.stat, num_runs)
