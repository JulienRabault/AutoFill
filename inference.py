import os
import yaml
import torch
import argparse
import logging
from model.VAE.pl_VAE import PlVAE
from dataset.datasetH5 import HDF5Dataset

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def inference(config):
    """
    Perform inference on 4 samples using a trained PlVAE model.
    """
    logger = logging.getLogger(__name__)
    ckpt_path = "/projects/pnria/julien/autofill/runs/grid_ag_saxs_beta0.001_etamin1e-06_ld64_bs128/epoch=46-step=15369.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logger.info(f"Loading model from checkpoint: {ckpt_path}")
    model = PlVAE.load_from_checkpoint(ckpt_path, config=config)
    model.eval()

    logger.info("Initializing dataset...")
    dataset = HDF5Dataset(
        hdf5_file=config["dataset"]["hdf5_file"],
        metadata_filters=config["dataset"]["metadata_filters"],
        conversion_dict_path=config["dataset"]["conversion_dict_path"],
        sample_frac=config["dataset"]["sample_frac"],
        transform=config["dataset"]["transform"],
        requested_metadata=config["dataset"]["requested_metadata"],
    )

    indices = [0, 1, 2, 3]
    logger.info(f"Sampling indices: {indices}")
    samples = [dataset[i] for i in indices]

    batch = {
        "data_q": torch.cat([s["data_q"] for s in samples], dim=0),
        "data_y": torch.cat([s["data_y"] for s in samples], dim=0),
        "metadata": [s["metadata"] for s in samples]
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch["data_q"] = batch["data_q"].to(device)
    batch["data_y"] = batch["data_y"].to(device)

    logger.info(f"Running inference on batch of size {len(samples)}...")
    with torch.no_grad():
        outputs = model(batch)

    logger.info("Inference completed.")
    for i, sample in enumerate(samples):
        logger.info(f"Sample {i} | CSV Index: {sample['csv_index']} | Length: {sample['len']}")
        logger.info(f"Metadata: {sample['metadata']}")
        logger.info(f"Output: {outputs[i].cpu()}")

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    inference(config)