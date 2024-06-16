#!/bin/bash
#SBATCH --job-name=harlow
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=48G
#SBATCH -e /home/sc10264/samplingrnn/template_harlow/results/slurm-%A_%a.err
#SBATCH -o /home/sc10264/samplingrnn/template_harlow/results/slurm-%A_%a.out
#SBATCH --array=0

python -u test.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=/home/sc10264/samplingrnn/template_harlow/results