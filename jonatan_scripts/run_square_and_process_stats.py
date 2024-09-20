import datetime
import os
import re
import shutil
import subprocess


def run_gem5():
    command = [
        "sudo",
        "build/VEGA_X86/gem5.opt",
        "configs/example/gpufs/mi300.py",
        "--disk-image",
        "gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml",
        "--kernel",
        "gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml",
        "--app",
        "gem5-resources/src/gpu/square/bin.default/square.default",
    ]
    subprocess.run(command, check=True)


def extract_gpu_stats(file_path):
    gpu_stats = []

    # Read and parse the stats file
    with open(file_path) as file:
        for line in file:
            if (
                "gpu" in line or "cpu1" in line
            ):  # cpu1 is the GPU, see system.py line ~108
                gpu_stats.append(line.strip())

    return gpu_stats


def write_stats(stats, file_path):
    with open(file_path, "w") as file:
        for line in stats:
            file.write(f"{line}\n")


def move_output_files():
    # Create the destination directory if it doesn't exist
    destination_dir = "jonatan_runs"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the new folder name with the timestamp
    new_folder_name = f"m5out_{timestamp}"

    # Rename and move the m5out folder
    shutil.move("m5out", os.path.join(destination_dir, new_folder_name))

    print(f"Moved m5out to {os.path.join(destination_dir, new_folder_name)}")


def main():
    # Run gem5 simulation
    run_gem5()

    # Extract stats from stats.txt
    gpu_stats = extract_gpu_stats("m5out/stats.txt")

    # Write extracted stats to a file
    write_stats(gpu_stats, "m5out/gpu_stats.txt")

    # Move the output files to a new folder
    move_output_files()


if __name__ == "__main__":
    main()
