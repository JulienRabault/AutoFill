#!/bin/sh
#SBATCH --job-name=AutoFillProject
#SBATCH --output=ML-%j.out
#SBATCH --error=ML-%j.err

#SBATCH --partition=RTX8000Nodes
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1  # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=4
#SBATCH --gres-flags=enforce-binding

echo "GPUs utilisés : $SLURM_JOB_GPUS"
export NCCL_DEBUG=INFO

srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif \
    /projects/pnria/julien/env/autofill_env/bin/python "trainpair.py"