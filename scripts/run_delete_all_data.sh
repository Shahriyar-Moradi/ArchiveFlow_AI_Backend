#!/bin/bash
# Script to run delete_all_data.py with conda llm10 environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm10

# Change to backend directory
cd "$BACKEND_DIR"

# Run the deletion script with all arguments passed through
python scripts/delete_all_data.py "$@"
