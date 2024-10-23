import argparse
import datetime
import os
import shutil
import subprocess

from dnnmark_list import BENCHMARKS
from extract_gpu_stats import extract_gpu_stats
from variants import VARIANTS, VARIANT_NAMES, get_variant


def run_gem5(binary, parameters, overrides):
    command = [
        "sudo",
        "build/VEGA_X86/gem5.opt",
        "configs/example/gpufs/mi300.py",
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
        choices=VARIANT_NAMES,
    )
    args = parser.parse_args()

    if args.variant:
        if variant := get_variant(args.variant) is None:
            print(f"Variant {args.variant} not found")
            exit(1)
        VARIANTS = [variant]

    for (variant, overides) in VARIANTS:
        for name, (binary, parameters) in BENCHMARKS.items():
            run_gem5(binary, parameters, overides)
            extract_gpu_stats("m5out/stats.txt", "m5out/gpu_stats.txt")
            move_output_files(name, timestamp, variant)
