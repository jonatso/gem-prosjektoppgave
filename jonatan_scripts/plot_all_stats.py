import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from dnnmark_list import BENCHMARKS
from get_single_stat import get_single_stat_all_runs
from variants import VARIANTS


def plot_all_stats(all_stats, stat_name, num_runs):
    benchmarks = list(all_stats.keys())
    variants = [variant for variant, _1, _2 in VARIANTS]

    N_benchmarks = len(benchmarks)
    N_variants = len(variants)

    bar_width = 0.8 / N_variants  # Width of each bar within a group
    group_width = N_variants * bar_width
    spacing = 0.2  # Spacing between benchmark groups

    x = np.arange(N_benchmarks) * (
        group_width + spacing
    )  # Positions for benchmark groups

    fig, ax = plt.subplots(figsize=(20, 5))  # Adjust figsize as needed

    # Calculate normalization factors for each benchmark based on the first variant
    norm_factors = {}
    first_variant = variants[0]
    for benchmark in benchmarks:
        avg, _ = all_stats[benchmark].get(first_variant, (None, None))
        if avg is not None and avg != 0:
            norm_factors[benchmark] = avg
        else:
            norm_factors[benchmark] = 1  # Avoid division by zero

    for i, variant in enumerate(variants):
        y = []
        yerr = []
        for benchmark in benchmarks:
            avg, std = all_stats[benchmark].get(variant, (None, None))
            norm_factor = norm_factors[benchmark]
            if avg is not None:
                y.append(avg / norm_factor)
                yerr.append(std / norm_factor)
            else:
                y.append(0)
                yerr.append(0)

        positions = x + i * bar_width
        ax.bar(
            positions,
            y,
            width=bar_width,
            yerr=yerr,
            capsize=5,
            label=variant,
        )

    ax.set_xlabel("Benchmark")
    ax.set_ylabel(f"Normalized {stat_name}")
    ax.set_title(
        f"{stat_name} for all benchmarks (Normalized to {first_variant}). Average of {num_runs} runs"
    )
    ax.set_xticks(x + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(benchmarks)

    ax.legend()

    plt.tight_layout()
    output_dir = f"jonatan_images/{stat_name}"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "all_benchmarks.svg")
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

    all_stats = {}
    num_runs = None

    for benchmark_name in BENCHMARKS.keys():
        stats, num_runs = get_single_stat_all_runs(benchmark_name, args.stat)
        all_stats[benchmark_name] = stats
        print(f"Stats for {benchmark_name}:")
        print(stats)

    plot_all_stats(all_stats, args.stat, num_runs)
