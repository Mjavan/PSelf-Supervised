#!/bin/bash

# Default values for arguments
DEFAULT_SCRIPT="core/bayesianbyol.py"
seed=42
exp=600
num_epochs=20



# Check if arguments are passed, otherwise use defaults
SCRIPT=${1:-$DEFAULT_SCRIPT}
ARG1=${2:-$seed}
ARG2=${3:-$exp}
ARG3=${4:-$num_epochs}



# Run the Python script with the arguments
python $SCRIPT --seed $ARG1 --exp $ARG2 --num_epochs $ARG3 