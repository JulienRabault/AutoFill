#!/bin/sh
#SBATCH --job-name=ping_mlflow
#SBATCH --output=log.err
#SBATCH --error=log.err

#SBATCH --partition=GPUNodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding


srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/julien/env/autofill_env/bin/python /projects/pnria/julien/autofill/test_mlflow.py