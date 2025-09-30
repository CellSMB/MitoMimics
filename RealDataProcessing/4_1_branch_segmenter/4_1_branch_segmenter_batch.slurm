#!/bin/bash

# Partition for the job:
#SBATCH --partition=interactive


#SBATCH --nodes=1


# The name of the job:
#SBATCH --job-name="branch_segmentation_final"

#SBATCH --account="punim1102"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --array=0-1

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
SAMPLES=(5)

SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}

echo "Processing Sample ${SAMPLE}"


# Run the job from the directory where it was launched (default)

# The modules to load:
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/venv-3.10.4-mito_branch_seg/bin/activate


# The job command(s):
python 4_1_branch_segmenter_hpc.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --sample ${SAMPLE} \
  --batch_size 5 \
  --max_workers 5



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
