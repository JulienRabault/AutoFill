#!/usr/bin/env python3
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def submit_job(mode, gridsearch, config, tech, mat, gpu):
    print(config)
    job_name = f"{mode}_{tech}_{mat}-%j"
    log_file = f"log_{mode}_tech_{tech}_mat_{mat}-%j.err"
    log_file2 = f"log_{mode}_tech_{tech}_mat_{mat}-%j.out"

    command = [
        "sbatch",
        "--job-name", job_name,
        "--output", log_file2,
        "--error", log_file,
        "--partition", "GPUNodes",
        "--nodes", "1",
        f"--gres=gpu:{gpu}",
        f"--ntasks-per-node","1",
        "--cpus-per-task", "8",
        "--gres-flags", "enforce-binding",
        "--wrap",
        f"srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/pnria/julien/env/autofill_env/bin/python train.py --name {tech}_{mat}_gpu{gpu} --mode {mode} --config {config} --gridsearch {gridsearch} --technique {tech} --material {mat} --devices {gpu}"
    ]

    logging.info(f"Submitting job: {job_name}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        logging.info(f"Job {job_name} submitted successfully.")
    else:
        logging.error(f"Failed to submit job {job_name}: {result.stderr}")
    subprocess.run(["squeue","--me"])

#python3 srun.py --mode pair_vae --mat ag --gpu 0  --gridsearch off
#python3 srun.py --mode vae --tech les --mat ag --gridsearch off 
#python3 srun.py --mode vae --tech les --mat ag --gridsearch on
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="Model à utiliser", choices=["vae", "pair_vae"])
    parser.add_argument("--gridsearch", required=True, help="Turn gridsearch on or off", choices=["on", "off"])
    parser.add_argument("--config", required=False, help="Chemin vers config de base à utiliser")
    parser.add_argument("--tech", required=False, help="Technique séparé par ','")
    parser.add_argument("--mat", required=False, help="Matériau séparé par ','")
    parser.add_argument("--gpu", default=1, type=int, help="Nombre de GPUs")


    args = parser.parse_args()

    if args.gridsearch == "on":
        if args.mode == "pair_vae":
            config = "model/pair_vae2.yaml"
        else :
            if args.tech == "saxs":
                config = "model/VAE/vae_config_saxs.yaml"
            elif args.tech == "les":
                config = "model/VAE/vae_config_les.yaml"
            else:
                raise ValueError("Tech")
                
    else :
        config = args.config
        if config is None :
            if args.mode == "pair_vae":
                config = "model/pair_vae2.yaml"
            else :
                if args.tech == "saxs":
                    config = "model/VAE/vae_config_saxs.yaml"
                elif args.tech == "les":
                    config = "model/VAE/vae_config_les.yaml"
                else:
                    raise ValueError("Tech")
    
    submit_job(args.mode, args.gridsearch, config, args.tech, args.mat, args.gpu)