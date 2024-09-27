"""
make a script that runs all the rodinia benchmarks
the output should be moved to a folder with a timestamp in jonatan_runs
the folder should be named m5out_<benchmark>_<timestamp>
Here is an example of how to run the kmeans benchmark:
sudo build/VEGA_X86/gem5.opt configs/example/gpufs/hip_rodinia.py --disk-image gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml --kernel gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml --app kmeans --gpu-mmio-trace gem5-resources/src/gpu-fs/vega_mmio.log
"""

import argparse
import datetime
import os
import shutil
import subprocess

from extract_gpu_stats import extract_gpu_stats
from rodinia_list import BENCHMARKS
from variants import VARIANTS


def run_gem5(benchmark, overrides):
    command = [
        "sudo",
        "build/VEGA_X86/gem5.opt",
        "configs/example/gpufs/hip_rodinia.py",
        "--disk-image",
        "gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml",
        "--kernel",
        "gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml",
        "--app",
        benchmark,
        "--gpu-mmio-trace",
        "gem5-resources/src/gpu-fs/vega_mmio.log",
    ] + (overrides if overrides else [])
    subprocess.run(command, check=True)


def move_output_files(benchmark, timestamp, variant):
    output_folder = f"jonatan_runs/{variant}/{benchmark}/{timestamp}"
    os.makedirs(output_folder)
    for filename in os.listdir("m5out"):
        shutil.move(os.path.join("m5out", filename), output_folder)
    os.rmdir("m5out")


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # variants is a dict, key is the variant name, value is a list of overrides
    # add variant as a input argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--variant",
        default=None,
        help="Variant to run",
        choices=VARIANTS.keys(),
    )
    args = parser.parse_args()

    if args.variant:
        if args.variant not in VARIANTS:
            print(f"Variant {args.variant} not found")
            exit(1)
        VARIANTS = {args.variant: VARIANTS[args.variant]}

    for variant, overides in VARIANTS.items():
        for benchmark in BENCHMARKS:
            run_gem5(benchmark, overides)
            extract_gpu_stats("m5out/stats.txt", "m5out/gpu_stats.txt")
            move_output_files(benchmark, timestamp, variant)
