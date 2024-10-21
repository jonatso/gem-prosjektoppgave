import argparse
import os

import matplotlib.pyplot as plt
from dnnmark_list import BENCHMARKS


def get_single_stat(input_file_path, stat_name):
    with open(input_file_path) as file:
        for line in file:
            if stat_name in line:
                # Assuming the format is "stat_name value # extra_info"
                parts = line.split()
                if len(parts) > 1 and parts[1] != "nan":
                    return float(parts[1])
    return None


def get_single_stat_all_runs(benchmark, stat_name, get_average=False):
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
    if get_average:
        for variant in res:
            res[variant] = sum(res[variant]) / len(res[variant])
    return res


def plot_stats(stats, benchmark_name, stat_name, get_average):
    plt.figure()
    for variant, stat_values in stats.items():
        if isinstance(stat_values, list):
            plt.plot(stat_values, label=variant)
        else:
            plt.bar(variant, stat_values, label=variant)

    plt.xlabel("Run Index" if not get_average else "Variant")
    # plt.ylabel(stat_name)
    plt.title(f"{stat_name} for {benchmark_name}")
    plt.legend()

    output_dir = f"jonatan_images/{stat_name}"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(
        output_dir, f"{benchmark_name}{'_average' if get_average else ''}.png"
    )
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
    parser.add_argument(
        "-a",
        "--average",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Get the average of the stats",
    )
    args = parser.parse_args()

    for benchmark_name in BENCHMARKS.keys():
        stats = get_single_stat_all_runs(
            benchmark_name, args.stat, args.average
        )
        print(f"Stats for {benchmark_name}:")
        print(stats)
        plot_stats(stats, benchmark_name, args.stat, args.average)
