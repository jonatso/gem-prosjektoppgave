import argparse
import os
from typing import (
    Dict,
    List,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
from dnnmark_list import BENCHMARKS
from get_single_stat import get_single_stat_all_runs
from variants import VARIANTS


def plot_all_stats(all_stats, stat_name, filename="all_benchmarks"):
    benchmarks = list(all_stats.keys())
    variants = [variant for variant, _1, _2 in VARIANTS]

    # Calculate num_runs from the first available data
    num_runs = None
    for benchmark in benchmarks:
        for variant in variants:
            runs_data = all_stats[benchmark].get(variant, (None, None))
            if runs_data is not None:
                num_runs = len(runs_data)
                break
        if num_runs is not None:
            break

    if num_runs is None:
        num_runs = 0  # or some default value

    N_benchmarks = len(benchmarks)
    N_variants = len(variants)

    bar_width = 0.8 / N_variants
    group_width = N_variants * bar_width
    spacing = 0.2

    x = np.arange(N_benchmarks) * (group_width + spacing)

    fig, ax = plt.subplots(figsize=(20, 5))

    # Calculate normalization factors based on the first variant
    norm_factors = {}
    first_variant = variants[0]
    for benchmark in benchmarks:
        runs_data = all_stats[benchmark].get(first_variant, 0)
        if runs_data is not None:
            # Calculate average across all runs
            avg = np.mean(runs_data)
            if avg != 0:
                norm_factors[benchmark] = avg
            else:
                norm_factors[benchmark] = 1
        else:
            norm_factors[benchmark] = 1

    for i, variant in enumerate(variants):
        y = []
        yerr = []
        for benchmark in benchmarks:
            runs_data = all_stats[benchmark].get(variant, 0)
            norm_factor = norm_factors[benchmark]

            if runs_data is not None:
                # Calculate mean and std across runs
                avg = np.mean(runs_data)
                std = np.std(runs_data)
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
    filename = os.path.join(output_dir, f"{filename}.svg")
    plt.savefig(filename, format="svg")
    plt.close()


def process_multi_values(all_stats: Dict) -> Dict[str, Dict]:
    """Process multiple values per run into different statistical measures."""
    processed_results = {
        "mean": {},
        "min": {},
        "max": {},
        "std_rel": {},
        "sum": {},
    }

    # For each benchmark
    for benchmark, variants_data in all_stats.items():
        # For each variant
        for measure in processed_results.keys():
            processed_results[measure][benchmark] = {}
            for variant, (values_per_run) in variants_data.items():
                if values_per_run is None:
                    processed_results[measure][benchmark][variant] = 0
                    continue

                # Calculate the statistic for each run
                run_stats = []
                for run_values in values_per_run:
                    if measure == "mean":
                        run_stats.append(np.mean(run_values))
                    elif measure == "min":
                        run_stats.append(np.min(run_values))
                    elif measure == "max":
                        run_stats.append(np.max(run_values))
                    elif measure == "std_rel":
                        mean = np.mean(run_values)
                        std = np.std(run_values)
                        run_stats.append(std / mean if mean != 0 else 0)
                    elif measure == "sum":
                        run_stats.append(np.sum(run_values))
                print(f"{measure} for {benchmark} {variant}")
                print(run_stats)

                # Calculate mean and std across runs
                processed_results[measure][benchmark][variant] = run_stats

    return processed_results


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
        "-m",
        "--multi-value",
        action="store_true",
        help="Whether the stat has multiple values per run",
    )
    args = parser.parse_args()

    all_stats = {}

    # Collect stats
    for benchmark_name in BENCHMARKS.keys():
        stats = get_single_stat_all_runs(
            benchmark_name, args.stat, args.multi_value
        )
        all_stats[benchmark_name] = stats

    # Process multi-value stats if needed
    if args.multi_value:
        print("Processing multi-value stats")
        processed_stats = process_multi_values(all_stats)
        # Create a plot for each statistical measure
        for measure, stats in processed_stats.items():
            plot_all_stats(stats, f"{args.stat} ({measure})", f"{measure}")
    else:
        plot_all_stats(all_stats, args.stat)
