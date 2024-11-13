import argparse
import os

import matplotlib.pyplot as plt
from dnnmark_list import BENCHMARKS
from get_single_stat import get_single_stat_all_runs


def plot_stats(stats, benchmark_name, stat_name, num_runs):
    plt.figure()
    variants = list(stats.keys())
    averages = [stats[variant][0] for variant in variants]
    std_devs = [stats[variant][1] for variant in variants]

    # Generate a list of colors for each variant
    colors = plt.get_cmap("tab10", len(variants)).colors

    plt.bar(
        variants,
        averages,
        yerr=std_devs,
        capsize=5,
        color=colors,
        label=f"Average of {num_runs} runs",
    )

    plt.xlabel("Variant")
    plt.ylabel(stat_name)
    plt.title(f"{stat_name} for {benchmark_name}")
    plt.xticks(rotation=90)  # Rotate x-axis labels 90 degrees
    plt.legend()

    # Adjust layout to make room for rotated x-axis labels
    plt.tight_layout()  # Automatically adjusts layout

    output_dir = f"jonatan_images/{stat_name}"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{benchmark_name}.svg")
    plt.savefig(filename, format="svg")
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
        stats, num_runs = get_single_stat_all_runs(benchmark_name, args.stat)
        print(f"Stats for {benchmark_name}:")
        print(stats)
        plot_stats(stats, benchmark_name, args.stat, num_runs)
