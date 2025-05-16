import abc
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.datasetH5 import HDF5Dataset
from src.dataset.datasetTXT import TXTDataset
from src.model.pairvae.pl_pairvae import PlPairVAE
from src.model.vae.pl_vae import PlVAE


class BaseInferencer:
    def __init__(self, checkpoint_path, data_path, hparams, conversion_dict_path=None, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.output_dir = "inference_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self.load_model(checkpoint_path)
        self.config = hparams
        self.model.to(self.device).eval()
        input_dim = self.config["model"]["args"]["input_dim"]
        self.compute_dataset(conversion_dict_path, data_path, input_dim)

    @abc.abstractmethod
    def compute_dataset(self, conversion_dict_path, data_path, input_dim):
        raise NotImplementedError(
            "compute_dataset method should be implemented in subclasses."
        )

    def load_model(self, path):
        raise NotImplementedError

    def infer_and_save(self):
        raise NotImplementedError

    def save_pred(self, batch, i, q_arr, y_arr):
        stacked = np.stack([y_arr, q_arr], axis=1)
        try:
            if self.format == 'h5':
                idx = batch['csv_index'][i]
                name = str(idx)
            else:
                path = batch['path'][i]
                name = os.path.splitext(os.path.basename(path))[0]
        except KeyError:
            # Handle the case where 'csv_index' or 'path' is not in the batch
            name = f"sample_{i}"
        filename = f"prediction_{name}.npy"
        np.save(os.path.join(self.output_dir, filename), stacked)

    def _move_to_device(self, batch):
        batch['data_y'] = batch['data_y'].to(self.device)
        batch['data_q'] = batch['data_q'].to(self.device)
        return batch

    def infer(self):
        self.infer_and_save()
        print(f"Inference results saved in {self.output_dir}")

    @abc.abstractmethod
    def infer_and_save(self):
        raise NotImplementedError("The infer_and_save method should be implemented in subclasses.")

class VAEInferencer(BaseInferencer):
    def load_model(self, path):
        return PlVAE.load_from_checkpoint(checkpoint_path=path)

    def infer_and_save(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference per sample"):
                batch = self._move_to_device(batch)
                outputs = self.model(batch)
                y_pred = outputs["recon"].squeeze(dim=0)
                q_pred = batch['data_q']
                for i in range(len(y_pred)):
                    y_arr = y_pred[i].cpu().numpy().flatten()
                    q_arr = q_pred[i].cpu().numpy().flatten()
                    self.save_pred(batch, i, q_arr, y_arr)


    def compute_dataset(self, conversion_dict_path, data_path, input_dim):
        if data_path.endswith(".h5"):
            self.dataset = HDF5Dataset(
                data_path,
                transform=self.config["dataset"]["transform"],
                metadata_filters=self.config["dataset"]["metadata_filters"],
                conversion_dict_path=conversion_dict_path
            )
            self.format = 'h5'
        elif data_path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(data_path)
            df = df[df["technique"].str.lower().isin([t.lower() for t in self.config["dataset"]["metadata_filters"]["technique"]])]
            df = df[df["material"].str.lower().isin([m.lower() for m in self.config["dataset"]["metadata_filters"]["material"]])]
            df = df.reset_index(drop=True)
            self.dataset = TXTDataset(
                dataframe=df,
                transform={
                    'y': {'MinMaxNormalizer': {}, 'PaddingTransformer': {'pad_size': input_dim, 'value': 0}},
                    'q': {'PaddingTransformer': {'pad_size': input_dim, 'value': 0}}
                }
            )
            self.format = 'csv'
        else:
            raise ValueError("Unsupported file format. Use .h5 or .csv")


class PairVAEInferencer(BaseInferencer):
    def __init__(self, checkpoint_path, data_path, mode, conversion_dict_path=None, batch_size=32):
        if mode not in {'les_to_saxs', 'saxs_to_les'}:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'les_to_saxs' or 'saxs_to_les'.")
        self.mode = mode
        super().__init__(checkpoint_path, data_path, conversion_dict_path, batch_size)

    def load_model(self, path):
        return PlPairVAE.load_from_checkpoint(path)

    def infer_and_save(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        for batch in tqdm(loader, desc="PairVAE inference"):
            batch = self._move_to_device(batch)
            if self.mode == 'les_to_saxs':
                y_pred, q_pred = self.model.les_to_saxs(batch)
            elif self.mode == 'saxs_to_les':
                y_pred, q_pred = self.model.saxs_to_les(batch)
            else:
                raise ValueError(f"Unknown inference mode: {self.mode}")

            for i in range(len(y_pred)):
                y_arr = y_pred[i].cpu().numpy().flatten()
                q_arr = q_pred[i].cpu().numpy().flatten()
                stacked = np.stack([y_arr, q_arr], axis=1)
                if self.format == 'h5':
                    name = str(batch['csv_index'][i])
                else:
                    path = batch['path'][i]
                    name = os.path.splitext(os.path.basename(path))[0]
                np.save(os.path.join(self.output_dir, f"prediction_{name}.npy"), stacked)
