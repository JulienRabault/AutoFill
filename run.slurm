#!/bin/sh
#SBATCH --job-name=ping_mlflow
#SBATCH --output=ping_mlflow.out
#SBATCH --error=ping_mlflow.err

#SBATCH --partition=GPUNodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding

srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif \
    curl -I --max-time 10 https://www.youtube.com/