#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

python3 inference.py "working_dir/input/$1"
