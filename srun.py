#!/usr/bin/env python3
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def submit_job(mode, config, tech, mat, gpu):
    print(config)
    job_name = f"{mode}_{tech}_{mat}-%j"
    log_file = f"log_{mode}_tech_{tech}_mat_{mat}-%j.err"

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
        f"srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/julien/env/autofill_env/bin/python train.py --name {tech}_{mat}_gpu{gpu} --mode {mode} --config {config} --technique {tech} --material {mat} --devices {gpu}"
    ]

    logging.info(f"Submitting job: {job_name}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        logging.info(f"Job {job_name} submitted successfully.")
    else:
        logging.error(f"Failed to submit job {job_name}: {result.stderr}")
    subprocess.run("squeue")

#python3 srun.py --mode vae --config "model/VAE/vae_config_les.yaml" --mat ag --tech les
#python3 srun.py --mode vae --config "model/VAE/vae_config_saxs.yaml" --mat ag --tech saxs
#python3 srun.py --mode pair_vae --config "model/pair_vae2.yaml" --mat ag 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="Model à utiliser")
    parser.add_argument("--config", required=True, help="Chemin vers config de base à utiliser")
    parser.add_argument("--tech", required=False, help="Technique séparé par ','")
    parser.add_argument("--mat", required=False, help="Matériau séparé par ','")
    parser.add_argument("--gpu", default=1, type=int, help="Nombre de GPUs")

    args = parser.parse_args()
    submit_job(args.mode, args.config, args.tech, args.mat, args.gpu)
