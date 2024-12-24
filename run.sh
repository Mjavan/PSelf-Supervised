#!/bin/bash

# Default values for arguments
DEFAULT_SCRIPT="core/bayesianbyol.py"
seed=42
exp=600
num_epochs=10
cycle_length=4
epoch_noise=2
epoch_st=0



# Check if arguments are passed, otherwise use defaults
SCRIPT=${1:-$DEFAULT_SCRIPT}
ARG1=${2:-$seed}
ARG2=${3:-$exp}
ARG3=${4:-$num_epochs}
ARG4=${5:-$cycle_length}
ARG5=${6:-$epoch_noise}
ARG6=${7:-$epoch_st}



# Run the Python script with the arguments
python $SCRIPT --seed $ARG1 --exp $ARG2 --num_epochs $ARG3 --cycle_length $ARG4 --epoch_noise $ARG5 --epoch_st $ARG6
