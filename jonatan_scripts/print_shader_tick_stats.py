from get_single_stat import get_single_stat

def print_shader_tick_stats(input_file_path):
    interesing_stat_names = [
        "simTicks",
        "shaderFirstActiveTick",
        "shaderLastActiveTick",
        "shaderActiveTicks",
        "shaderNonActiveInBetweenTicks",
    ]

    stats = {}
    for stat_name in interesing_stat_names:
        stats[stat_name] = get_single_stat(input_file_path, stat_name)
        print(f"{stat_name}: {stats[stat_name]}")

    print()
    print("Andel av tid som shader er aktiv (i den tiden der arbeider/får nytt arbeid):")
    print(f"{stats['shaderActiveTicks'] / (stats['shaderActiveTicks'] + stats['shaderNonActiveInBetweenTicks'])} : shaderActiveTicks / (shaderActiveTicks + shaderNonActiveInBetweenTicks)")

    print()
    print("Andel av tid som shader er aktiv eller venter på nytt arbeid")
    print(f"{(stats['shaderActiveTicks'] + stats['shaderNonActiveInBetweenTicks']) / stats['simTicks']} : (shaderActiveTicks + shaderNonActiveInBetweenTicks) / simTicks")

    print()
    print("Andel av tid som shader er aktiv (total):")
    print(f"{stats['shaderActiveTicks'] / stats['simTicks']} : shaderActiveTicks / simTicks")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Bruk: python3 print_shader_tick_stats.py <input_file_path>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    print_shader_tick_stats(input_file_path)
