#!/bin/bash

total=0
count=0

while IFS= read -r -d '' file; do
    # Count rows excluding header
    lines=$(($(wc -l < "$file") - 1))
    total=$((total + lines))
    count=$((count + 1))
done < <(find input -type f -name "*.csv" -print0)

if [ $count -gt 0 ]; then
    average=$(awk "BEGIN {printf \"%.2f\", $total/$count}")
    echo "CSV files found: $count"
    echo "Average rows (excluding header): $average"
else
    echo "No CSV files found"
fi
