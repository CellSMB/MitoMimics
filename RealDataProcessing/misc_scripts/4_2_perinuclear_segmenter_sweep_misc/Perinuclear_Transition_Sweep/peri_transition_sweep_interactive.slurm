#!/bin/bash

# Partition for the job:
#SBATCH --partition=interactive


#SBATCH --nodes=1


# The name of the job:
#SBATCH --job-name="peri_transition_overlap_sweep"

#SBATCH --account="punim1102"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --array=0-29

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-00:25:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Define data directories
DATA_DIR="input_data/"
OUTPUT_DIR="processed_outputs/"

# Define samples (modify as needed for your dataset)
SAMPLES=(7 12 13)
NUM_SAMPLES=${#SAMPLES[@]}

# Define threshold values
THRESH_VALUES=(1.2)
NUM_THRESHOLDS=${#THRESH_VALUES[@]}

# Define Peri Transition Threshold Value
PERI_OVERLAP=(0.5 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
NUM_PERI=${#PERI_OVERLAP[@]}

# Calculate which sample and threshold to use based on array task ID
TOTAL_PERI_THRESH=$((NUM_THRESHOLDS * NUM_PERI))
SAMPLE_IDX=$((SLURM_ARRAY_TASK_ID / TOTAL_PERI_THRESH))
REMAINDER=$((SLURM_ARRAY_TASK_ID % TOTAL_PERI_THRESH))
THRESH_IDX=$((REMAINDER / NUM_PERI))
PERI_IDX=$((SLURM_ARRAY_TASK_ID % NUM_PERI))

SAMPLE=${SAMPLES[$SAMPLE_IDX]}
THRESH=${THRESH_VALUES[$THRESH_IDX]}
PERI=${PERI_OVERLAP[$PERI_IDX]}

echo "Processing Sample ${SAMPLE} with threshold multiplier ${THRESH} and perinuclear transition overlap ratio ${PERI}"


# Run the job from the directory where it was launched (default)

# The modules to load:
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/venv-3.10.4-mito_peri_seg/bin/activate


# The job command(s):
python 1_peri_segmentation_sweep_peri_transition_sweep_hpc.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --sample ${SAMPLE} \
  --thresh_adj_mult ${THRESH} \
  --sigma 10 \
  --min_obj_size 5000 \
  --transition_thresh ${PERI} \
  --overlap_thresh 0.05 \
  --window_size 10 \
  --avg_threshold 0.5 \
  --max_workers 4



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
