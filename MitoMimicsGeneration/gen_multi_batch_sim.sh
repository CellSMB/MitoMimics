#!/bin/bash


# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 BATCH_SIZE NUMBER_BATCHES INITIAL_SEED"
    exit 1
fi

# Assign input arguments to variables
BATCH_SIZE=$1
NUMBER_BATCHES=$2
INITIAL_SEED=$3

# Function to run a batch with a specific number of iterations
run_batch() {
    START_SEED=$1
    END_SEED=$(( START_SEED + NUMBER_BATCHES - 1 ))
    for SEED in $(seq $START_SEED $END_SEED)
    do
        # Call the Python script with the current seed
        python sim.py --seed $SEED  # Use the script name from the argument
        wait  # Ensure each sim.py call completes before starting the next one
    done
}

# Run specified number of batches in the background
for (( i=0; i<BATCH_SIZE; i++ ))
do
    CURRENT_SEED=$(( INITIAL_SEED + i * NUMBER_BATCHES ))
    run_batch $CURRENT_SEED &
done

# Wait for all background jobs to finish


wait
