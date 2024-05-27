#!/bin/bash
#SBATCH --job-name=harlow
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=48G
# #SBATCH --gres=gpu:1
#SBATCH -e /home/sc10264/samplingrnn/template_harlow/results/slurm-%A_%a.err
#SBATCH -o /home/sc10264/samplingrnn/template_harlow/results/slurm-%A_%a.out
#SBATCH --array=0

# # Function to log GPU usage
# log_gpu_usage() {
#     while true; do
#         nvidia-smi >> /home/sc10264/samplingrnn/template_harlow/results/gpu_usage_$SLURM_ARRAY_JOB_ID.log
#         sleep 60
#     done
# }

# # Start logging GPU usage in the background
# log_gpu_usage &

# Run the script
python -u training_batch.py --jobid=$SLURM_ARRAY_TASK_ID --path=/home/sc10264/samplingrnn/template_harlow/results

# # Kill the background logging process
# kill %1