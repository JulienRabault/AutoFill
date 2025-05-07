import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset.datasetH5 import HDF5Dataset
from src.model.vae.pl_vae import PlVAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=False)
    return parser.parse_args()

class BaseRunner:
    def __init__(self, checkpoint_path, output_dir, h5_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self.load_model(checkpoint_path)
        self.model.to(self.device).eval()
        self.batch_size = 64
        self.h5_path = h5_path

    def load_model(self, path):
        raise NotImplementedError

    def build_loader(self):
        raise NotImplementedError

    def infer(self, loader):
        raise NotImplementedError

    def save(self, arr, name):
        np.save(os.path.join(self.output_dir, f"{name}.npy"), arr)

    def run(self):
        loader = self.build_loader()
        preds = self.infer(loader)
        for k, v in preds.items():
            self.save(v, k)
        print(f"Results saved to {self.output_dir}, for {len(loader.dataset)} samples.")

class VAEInfer(BaseRunner):
    def load_model(self, path):
        model = PlVAE.load_from_checkpoint(path)
        return model

    def build_loader(self):
        test_h5 = self.h5_path or self.model.hparams.data.test_hdf5
        input_dim = self.model.hparams.model.args.input_dim
        dataset = HDF5Dataset(
            test_h5,
            transform={
                'y': {
                    'MinMaxNormalizer': {},
                    'PaddingTransformer': {
                        'pad_size': input_dim,
                        'value': 0
                    }
                }
            }
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def infer(self, loader):
        results = []
        with torch.no_grad():
            for x in tqdm(loader, desc="VAE Inference"):
                x = x.to(self.device)
                out = self.model(x)
                tensor = out[0] if isinstance(out, (list, tuple)) else out
                results.append(tensor.cpu().numpy())
        return {"reconstruction": np.concatenate(results, axis=0)}

if __name__ == "__main__":
    args = parse_args()
    runner = VAEInfer(args.checkpoint, args.output, args.h5_path)
    runner.run()
