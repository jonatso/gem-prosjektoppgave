import numpy as np
from get_single_stat import get_stat


def print_max_min_avg_std(stat_name, stat_list):
    mean = sum(stat_list) / len(stat_list)
    print(
        f"{stat_name}: {mean:.4f} | max: {max(stat_list):.4f} | min: {min(stat_list):.4f} | rstd: {(np.std(stat_list) / mean * 100):.4f}% | num: {len(stat_list)}"
    )


def print_shader_tick_stats(input_file_path):
    interesting_single_stat_names = [
        r".*simTicks",
        r".*shaderFirstActiveTick",
        r".*shaderLastActiveTick",
        r".*shaderActiveTicks",
        r".*shaderNonActiveInBetweenTicks",
    ]

    interesting_multi_stat_names = [
        r".*CUs[0-9]*\.totalCycles",
        r".*CUs[0-9]*\.numInstrExecuted",
        r".*CUs[0-9]*\.ipc",
        r".*CUs[0-9]*\.numCuSleeps",
        r".*CUs[0-9]*\.wavefronts[0-9]*\.numInstrExecuted",
    ]

    single_stats = {}
    for stat_name in interesting_single_stat_names:
        single_stats[stat_name] = get_stat(input_file_path, stat_name)
        print(f"{stat_name}: {single_stats[stat_name]}")

    multi_stats = {}
    for stat_name in interesting_multi_stat_names:
        multi_stats[stat_name] = get_stat(
            input_file_path, stat_name, return_all=True
        )
        print_max_min_avg_std(stat_name, multi_stats[stat_name])

    print()
    print(
        "Andel av tid som shader er aktiv (i den tiden der arbeider/får nytt arbeid):"
    )
    print(
        f'{single_stats[r".*shaderActiveTicks"] / (single_stats[r".*shaderActiveTicks"] + single_stats[r".*shaderNonActiveInBetweenTicks"])} : shaderActiveTicks / (shaderActiveTicks + shaderNonActiveInBetweenTicks)'
    )

    print()
    print("Andel av tid som shader er aktiv eller venter på nytt arbeid")
    print(
        f'{(single_stats[r".*shaderActiveTicks"] + single_stats[r".*shaderNonActiveInBetweenTicks"]) / single_stats[r".*simTicks"]} : (shaderActiveTicks + shaderNonActiveInBetweenTicks) / simTicks'
    )

    print()
    print("Andel av tid som shader er aktiv (total):")
    print(
        f'{single_stats[r".*shaderActiveTicks"] / single_stats[r".*simTicks"]} : shaderActiveTicks / simTicks'
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Bruk: python3 print_shader_tick_stats.py <input_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    print_shader_tick_stats(input_file_path)
