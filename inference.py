import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetPairH5 import PairHDF5Dataset
from src.model.vae.pl_vae import PlVAE
from src.model.pairvae.pl_pairvae import PlPairVAE

class BaseInferenceRunner:
    def __init__(self, config, checkpoint_path, device):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.output_dir = config.get("output_dir", "inference_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        raise NotImplementedError

    def _build_dataloader(self):
        raise NotImplementedError

    def _save(self, array, name):
        path = os.path.join(self.output_dir, f"{name}.npy")
        np.save(path, array)

    def run(self):
        loader = self._build_dataloader()
        preds = self._infer(loader)
        for key, arr in preds.items():
            self._save(arr, key)

    def _infer(self, loader):
        raise NotImplementedError

class VAEInferenceRunner(BaseInferenceRunner):
    def _load_model(self):
        return PlVAE.load_from_checkpoint(self.checkpoint_path, hparams_file=self.config_path)

    def _build_dataloader(self):
        ds = HDF5Dataset(
            self.config["data"]["test_hdf5"],
            transform=self.config["data"].get("transform", {})
        )
        return DataLoader(ds, batch_size=self.config.get("batch_size", 1), shuffle=False)

    def _infer(self, loader):
        result = []
        with torch.no_grad():
            for x in tqdm(loader, desc="VAE Inference"):
                x = x.to(self.device)
                out = self.model(x)
                tensor = out[0] if isinstance(out, (list, tuple)) else out
                result.append(tensor.cpu().numpy())
        return {"reconstruction": np.concatenate(result, axis=0)}

class PairVAEInferenceRunner(BaseInferenceRunner):
    def _load_model(self):
        return PlPairVAE.load_from_checkpoint(self.checkpoint_path, hparams_file=self.config_path)

    def _build_dataloader(self):
        cfg = self.config["data"]
        ds = PairHDF5Dataset(
            hdf5_file_1=cfg["test_hdf5_1"],
            hdf5_file_2=cfg["test_hdf5_2"],
            conversion_dict_path=cfg["conversion_dict_path"],
            transform=cfg.get("transform", {})
        )
        return DataLoader(ds, batch_size=self.config.get("batch_size", 1), shuffle=False)

    def _infer(self, loader):
        preds = {"recon_saxs": [], "recon_les": [], "recon_saxs2les": [], "recon_les2saxs": []}
        with torch.no_grad():
            for batch in tqdm(loader, desc="PairVAE Inference"):
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                out = self.model(batch)
                for key in preds:
                    preds[key].append(out[key].cpu().numpy())
        return {k: np.concatenate(v, axis=0) for k, v in preds.items()}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_type = config["model"]["type"]
    runner_cls = VAEInferenceRunner if model_type == "VAE" else PairVAEInferenceRunner
    runner = runner_cls(config, args.checkpoint, device)
    runner.config_path = args.config
    runner.run()

if __name__ == "__main__":
    main()
