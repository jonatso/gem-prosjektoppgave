import datetime
import os
import re
import shutil
import subprocess


def run_gem5():
    command = [
        "sudo",
        "build/VEGA_X86/gem5.opt",
        "configs/example/gpufs/mi200.py",
        # "--gpu-mmio-trace",
        # "gem5-resources/src/gpu-fs/vega_mmio.log",
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

    # Define regex patterns for the desired formats
    patterns = [
        re.compile(r"system\.gpu_mem_ctrls.*"),
        re.compile(r"system\.mem_ctrls[0-1].*"),
        re.compile(r"system\.mem_ctrls1.*"),
        re.compile(r"system\.ruby\.network_gpu.*"),
        re.compile(r"system\.(l1|l2|l3)_tlb.*"),
        re.compile(r"system\.ruby\.tcp_cntrl[0-9]+\.L1Cache.*"),
        re.compile(r"system\.ruby\.tcp_cntrl[0-9]+\.L2Cache.*"),
        re.compile(r"system\.ruby\.cp_cntrl0\.L1D0cache.*"),
        re.compile(r"system\.ruby\.cp_cntrl0\.L1Icache.*"),
        re.compile(r"system\.ruby\.cp_cntrl0\.L2cache.*"),
        re.compile(r"system\.ruby\.dir_cntrl0\.L3CacheMemory.*"),
        re.compile(r"system\.ruby\.gpu_dir_cntrl0\.L3CacheMemory.*"),
        re.compile(r"system\.ruby\.tcc_cntrl0\.L2cache.*"),
        re.compile(r"system\.(l1|l2|l3)_coalescer.*"),
    ]

    # Read and parse the stats file
    with open(file_path) as file:
        for line in file:
            for pattern in patterns:
                if pattern.match(line):
                    gpu_stats.append(line.strip())
                    break

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

    # # Extract stats from stats.txt
    # gpu_stats = extract_gpu_stats("m5out/stats.txt")

    # # Write extracted stats to a file
    # write_stats(gpu_stats, "m5out/gpu_stats.txt")

    # # Move the output files to a new folder
    # move_output_files()


if __name__ == "__main__":
    main()
