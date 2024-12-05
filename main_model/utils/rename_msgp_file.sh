#!/bin/bash

# Check if exactly two arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <string>"
    exit 1
fi

# Assign inputs to variables
input_folder=$1
input_string=$2

# Find and rename .msgp files
find "$input_folder" -type f -name '*.msgp' | while read -r file; do
    dir=$(dirname "$file")
    base=$(basename "$file")
    mv "$file" "${dir}/${input_string}_${base}"
done

echo "Renaming complete."
