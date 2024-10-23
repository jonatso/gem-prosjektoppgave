VARIANTS = [
    ("1-CU", ["--num-compute-units", "1"],),
    ("2-CUs", ["--num-compute-units", "2"],),
    ("4-CUs", ["--num-compute-units", "4"],),
    ("8-CUs", ["--num-compute-units", "8"],),
    ("16-CUs", ["--num-compute-units", "16"],),
    ("32-CUs", ["--num-compute-units", "32"],),
    ("64-CUs", ["--num-compute-units", "64"],),
]

VARIANT_NAMES = [name for name, _ in VARIANTS]

def get_variant(variant_name):
    for name, overrides in VARIANTS:
        if name == variant_name:
            return overrides
    return None