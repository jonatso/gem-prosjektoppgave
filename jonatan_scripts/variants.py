VARIANTS = [
    (
        "1-CU_MI300",
        "configs/example/gpufs/mi300.py",
        ["--num-compute-units", "1"],
    ),
    (
        "2-CUs_MI300",
        "configs/example/gpufs/mi300.py",
        ["--num-compute-units", "2"],
    ),
    (
        "4-CUs_MI300",
        "configs/example/gpufs/mi300.py",
        ["--num-compute-units", "4"],
    ),
    (
        "8-CUs_MI300",
        "configs/example/gpufs/mi300.py",
        ["--num-compute-units", "8"],
    ),
    (
        "16-CUs_MI300",
        "configs/example/gpufs/mi300.py",
        ["--num-compute-units", "16"],
    ),
    (
        "32-CUs_MI300",
        "configs/example/gpufs/mi300.py",
        ["--num-compute-units", "32"],
    ),
    (
        "64-CUs_MI300",
        "configs/example/gpufs/mi300.py",
        ["--num-compute-units", "64"],
    ),
    (
        "1-CU_MI200",
        "configs/example/gpufs/mi200.py",
        ["--num-compute-units", "1"],
    ),
    (
        "2-CUs_MI200",
        "configs/example/gpufs/mi200.py",
        ["--num-compute-units", "2"],
    ),
    (
        "4-CUs_MI200",
        "configs/example/gpufs/mi200.py",
        ["--num-compute-units", "4"],
    ),
    (
        "8-CUs_MI200",
        "configs/example/gpufs/mi200.py",
        ["--num-compute-units", "8"],
    ),
    (
        "16-CUs_MI200",
        "configs/example/gpufs/mi200.py",
        ["--num-compute-units", "16"],
    ),
    (
        "32-CUs_MI200",
        "configs/example/gpufs/mi200.py",
        ["--num-compute-units", "32"],
    ),
    (
        "64-CUs_MI200",
        "configs/example/gpufs/mi200.py",
        ["--num-compute-units", "64"],
    ),
]

VARIANT_NAMES = [name for name, _1, _2 in VARIANTS]


def get_variant(variant_name):
    for name, run_script, overrides in VARIANTS:
        if name == variant_name:
            return (name, run_script, overrides)
    return None
