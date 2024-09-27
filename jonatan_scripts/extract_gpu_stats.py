import re


def extract_gpu_stats(input_file_path, output_file_path):
    # Maybe GPU related stats I've found in the stats.txt file, can be expanded?

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

    # Open the output file for writing
    with open(output_file_path, "w") as output_file:
        # Read and parse the stats file
        with open(input_file_path) as input_file:
            for line in input_file:
                for pattern in patterns:
                    if pattern.match(line):
                        output_file.write(line)
                        break
