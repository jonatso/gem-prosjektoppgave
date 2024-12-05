import argparse
import datetime
import os
import shutil
import subprocess

from dnnmark_list import BENCHMARKS
from variants import (
    VARIANT_NAMES,
    VARIANTS,
    get_variant,
)

GEM5_OUTPUT_DIR = "m5out"
MY_OUTPUT_DIR = "jonatan_runs"


def run_gem5(
    binary,
    parameters,
    run_script,
    overrides,
    debug_flags=None,
):
    command = [
        "sudo",
        "build/VEGA_X86/gem5.opt",
        f"--debug-flags={debug_flags}" if debug_flags else "",
        "--debug-file=debug.out.gz",
        run_script,
        "--disk-image",
        "gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml",
        "--kernel",
        "gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml",
        "--app",
        "gem5-resources/src/gpu/DNNMark/jonatan_run_dnnmark.sh",
        "--opts",
        f"{binary} '{parameters}'",  # passed to the script, which will simply call $1 $2 after setting up the environment
    ] + (overrides if overrides else [])
    subprocess.run(command, check=True)


def move_output_files(benchmark, timestamp, variant, path_prefix):
    if path_prefix:
        output_folder = (
            f"{MY_OUTPUT_DIR}/{path_prefix}/{variant}/{benchmark}/{timestamp}"
        )
    else:
        output_folder = f"{MY_OUTPUT_DIR}/{variant}/{benchmark}/{timestamp}"
    os.makedirs(output_folder)
    for filename in os.listdir(GEM5_OUTPUT_DIR):
        shutil.move(os.path.join(GEM5_OUTPUT_DIR, filename), output_folder)
    os.rmdir(GEM5_OUTPUT_DIR)


if __name__ == "__main__":
    # variants is a dict, key is the variant name, value is a list of overrides
    # add variant as a input argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--variant",
        default=None,
        help="Variant to run",
        choices=VARIANT_NAMES,
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        default=None,
        help="Benchmark to run",
        choices=BENCHMARKS.keys(),
    )
    parser.add_argument(
        "-p",
        "--path_prefix",
        default=None,
        help="Path prefix for the output folder",
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        default=1,
        type=int,
        help="Number of runs",
    )
    parser.add_argument(
        "-d",
        "--debug-flags",
        default=None,
        help="Debug flags",
    )
    args = parser.parse_args()

    if args.variant:
        if (
            variant := get_variant(args.variant)
        ) is None:  # get whole variant, not only the name
            print(f"Variant {args.variant} not found")
            exit(1)
        VARIANTS = [variant]

    if args.benchmark:
        if args.benchmark not in BENCHMARKS:
            print(f"Benchmark {args.benchmark} not found")
            exit(1)
        BENCHMARKS = {args.benchmark: BENCHMARKS[args.benchmark]}

    for i in range(args.num_runs):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for variant, run_script, overides in VARIANTS:
            for name, (binary, parameters) in BENCHMARKS.items():
                run_gem5(
                    binary, parameters, run_script, overides, args.debug_flags
                )
                move_output_files(name, timestamp, variant, args.path_prefix)
