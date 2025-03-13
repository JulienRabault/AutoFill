#!/usr/bin/env python3
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def submit_job(tech, mat, gpu):
    job_name = f"n{gpu}_{tech}_{mat}"
    log_file = f"log_n_gpu{gpu}_{tech}_{mat}.err"

    command = [
        "sbatch",
        "--job-name", job_name,
        "--output", log_file,
        "--error", log_file,
        "--partition", "GPUNodes",
        "--nodes", "1",
        f"--gres=gpu:{gpu}",
        f"--ntasks-per-node={gpu}",
        "--cpus-per-task", "4",
        "--gres-flags", "enforce-binding",
        "--wrap",
        f"srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/julien/env/autofill_env/bin/python main.py --name {tech}_{mat}_gpu{gpu} --technique {tech} --material {mat} --devices {gpu}"
    ]

    logging.info(f"Submitting job: {job_name}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        logging.info(f"Job {job_name} submitted successfully.")
    else:
        logging.error(f"Failed to submit job {job_name}: {result.stderr}")
    subprocess.run("squeue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tech", required=True, help="Technique à utiliser")
    parser.add_argument("--mat", required=True, help="Matériau ou dataset")
    parser.add_argument("--gpu", default=1, type=int, help="Nombre de GPUs")

    args = parser.parse_args()
    submit_job(args.tech, args.mat, args.gpu)
