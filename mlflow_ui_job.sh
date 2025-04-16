#!/bin/bash
#SBATCH --job-name=mlflow-ui
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=mlflow_ui.log
#SBATCH --partition=48CPUNodes


export GUNICORN_CMD_ARGS="--timeout 180"

PORT=5000
HOSTNAME=$(hostname)

echo "MLflow UI sur http://$HOSTNAME:$PORT"

srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif \
    /projects/pnria/julien/env/autofill_env/bin/mlflow ui \
    --backend-store-uri /projects/pnria/julien/autofill/runs/mlrun --port $PORT --host 0.0.0.0


