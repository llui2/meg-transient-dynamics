#!/bin/bash
# Clean data folder but keep .csv files

DATA_DIR="$(dirname "$0")/data"

if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR"
    exit 1
fi

echo "Cleaning $DATA_DIR (keeping .csv files)..."

# Remove all subject directories (numeric folders)
find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Remove any non-csv files at root level
find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type f ! -name "*.csv" -delete

echo "Done. Remaining files:"
ls -la "$DATA_DIR"
