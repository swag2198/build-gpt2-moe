#!/bin/bash
#SBATCH -J single_gpu              # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --time=3-00:00             # Runtime in D-HH:MM (3 days)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=120G                 # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --partition=h100-ferranti   # Which partition will run your job
#SBATCH --output=/weka/bethge/mwe102/brendel/build-gpt2/slurmoutputs/myjob-%j.out      # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/weka/bethge/mwe102/brendel/build-gpt2/slurmoutputs/myjob-%j.err       # File to which STDERR will be written - make sure this is not on $HOME

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi

# Setup Phase
echo "Setting up environment..."
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate /weka/bethge/mwe102/.conda/llm

# this script is run from inside the scripts/ directory, go back to main directory
cd /weka/bethge/mwe102/brendel/build-gpt2/ # cd into your working space
pwd

# Display Python and environment info for debugging
which python
python --version
pip list | grep torch

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"

# arguments
echo "B: $1"
echo "seed: $2"

# Compute Phase
srun python train_single_gpu.py --B $1 --seed $2 --use_noisy_top_k --use_aux_loss --use_router_z_loss --router_use_full_prec --use_switch_tfm_init

echo "Finished at: $(date)"