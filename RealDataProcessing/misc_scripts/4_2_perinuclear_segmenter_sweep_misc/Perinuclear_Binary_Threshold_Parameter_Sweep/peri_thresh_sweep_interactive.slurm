#!/bin/bash

# Partition for the job:
#SBATCH --partition=interactive


#SBATCH --nodes=1


# The name of the job:
#SBATCH --job-name="peri_mask_thresh_sweep"

#SBATCH --account="punim1102"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-447

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-00:15:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Define data directories
DATA_DIR="input_data/"
OUTPUT_DIR="processed_outputs/"

# Define samples (modify as needed for your dataset)
SAMPLES=(1 2 3 4 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
NUM_SAMPLES=${#SAMPLES[@]}

# Define threshold values
THRESH_VALUES=(0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
NUM_THRESHOLDS=${#THRESH_VALUES[@]}

# Calculate which sample and threshold to use based on array task ID
SAMPLE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_THRESHOLDS))
THRESH_IDX=$((SLURM_ARRAY_TASK_ID % NUM_THRESHOLDS))

SAMPLE=${SAMPLES[$SAMPLE_IDX]}
THRESH=${THRESH_VALUES[$THRESH_IDX]}

echo "Processing Sample ${SAMPLE} with threshold multiplier ${THRESH}"


# Run the job from the directory where it was launched (default)

# The modules to load:
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/venv-3.10.4-mito_peri_seg/bin/activate


# The job command(s):
python 1_peri_segmentation_sweep_hpc.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --sample ${SAMPLE} \
  --thresh_adj_mult ${THRESH} \
  --sigma 10 \
  --min_obj_size 5000 \
  --window_size 10 \
  --avg_threshold 0.5 \
  --max_workers 4



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
