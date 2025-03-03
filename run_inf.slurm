#!/bin/sh
#SBATCH --job-name=AUTOFILL
#SBATCH --output=log_infer.err
#SBATCH --error=log_infer.err

#SBATCH --partition=RTX8000Nodes
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=4
#SBATCH --gres-flags=enforce-binding


srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif \
    /projects/pnria/julien/env/autofill_env/bin/python inference.py --checkpoint_path runs/les_ag_gpu1/version_0/checkpoints/vae-epoch=49-val_loss=0.00024.ckpt  --csv_log_dir runs/les_ag_gpu1/version_1
