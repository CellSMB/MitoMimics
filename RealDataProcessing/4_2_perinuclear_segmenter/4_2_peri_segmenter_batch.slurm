#!/bin/bash

# Partition for the job:
#SBATCH --partition=interactive


#SBATCH --nodes=1


# The name of the job:
#SBATCH --job-name="perinuclear_segmentation_final"

#SBATCH --account="punim1102"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --array=0-42

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-00:40:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Define data directories
DATA_DIR="input_data/"
OUTPUT_DIR="processed_outputs/"

# Define samples and associated threshold values to process
SAMPLES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
THRESH_VALUES=(1.5 1.3 1.3 1.4 1.4 0.8 1.2 1.1 1.1 1.0 1.2 1.2 1.2 1.3 1.1 1.1 1.2 1.1 1.1 1.0 1.6 1.5 1.6 1.4 1.2 1.3 1.3 1.3 1.4 1.9 1.5 1.3 1.3 1.2 1.2 1.3 1.3 1.4 1.2 1.4 1.5 1.4 1.5)

SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
THRESH=${THRESH_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Processing Sample ${SAMPLE} with threshold multiplier ${THRESH}"


# Run the job from the directory where it was launched (default)

# The modules to load:
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/venv-3.10.4-mito_peri_seg/bin/activate


# The job command(s):
python 4_2_peri_segmenter_hpc.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --sample ${SAMPLE} \
  --thresh_adj_mult ${THRESH} \
  --sigma 10 \
  --min_obj_size 5000 \
  --transition_thresh 0.65 \
  --overlap_thresh 0.05 \
  --window_size 10 \
  --max_workers 4



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
