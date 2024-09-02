#!/bin/bash

#=================================================================
#
# Job script for running a job on a single GPU (any available GPU)
#
#=================================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the gpu partition (queue) with any GPU
#SBATCH --partition=gpu --gres=gpu:A100 --mem-per-cpu=9600
#
# Specify project account (replace as required)
#SBATCH --account=moshfeghi-pmwc
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=01:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
# Job name
#SBATCH --job-name=gpu_test
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

#Uncomment the following if you are running multi-threaded
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#
#=========================================================
# Prologue script to record job details
# Do not change the line below
#=====================
#----------------------------------------------
#Modify the line below to run your program. This is an example
#=========================================================

python /users/gxb18167/EEG-To-Text-Original/eval_decoding.py \
    --checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf True -n False
# Do not change the line below
#=========================================================sbat
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
