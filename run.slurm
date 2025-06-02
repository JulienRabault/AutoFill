#!/bin/sh
#SBATCH --job-name=ping_mlflow
#SBATCH --output=log.err
#SBATCH --error=log.err

#SBATCH --partition=GPUNodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding


srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/caroline/AUTOFILL/env_autofill/bin/python script/05_infer.py \
-c /projects/pnria/julien/autofill/AUTOFILL_LES/les_cylinder/best.ckpt \
-d /projects/pnria/caroline/test.h5 \
-o /projects/pnria/caroline/AUTOFILL/test_inference \
-s 0.0001 --plot