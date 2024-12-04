#!/bin/bash
# find the newest file named stats.txt in the jonatan_runs directory, print creation date and open it with code

if [ -z "$1" ]; then
    filename="stats.txt"
else
    filename="$1"
fi

newest_file=$(find jonatan_runs -name "$filename" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
echo $newest_file
echo $(date -r $newest_file)
code $newest_file
