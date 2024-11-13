import argparse

from dnnmark_list import BENCHMARKS
from get_single_stat import get_single_stat_all_runs
from variants import VARIANT_NAMES

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

    first_variant = VARIANT_NAMES[0]

    for benchmark_name in BENCHMARKS.keys():
        stats, _ = get_single_stat_all_runs(benchmark_name, args.stat)
        avg_first_variant = stats[first_variant][0]
        print(f"Average for {benchmark_name}:")
        print(avg_first_variant)
