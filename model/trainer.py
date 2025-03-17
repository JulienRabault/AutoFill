import os
import shutil
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from model.pl_PairVAE import PlPairVAE
from dataset.datasetPairH5 import PairHDF5Dataset
from lightning.pytorch.callbacks import ModelCheckpoint

def train(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("========================================")
    print("INIT Model")
    model = PlPairVAE(config)
    print("========================================")

    print("========================================")
    print("INIT Dataset")
    dataset = PairHDF5Dataset(
        hdf5_file=config["dataset"]["hdf5_file"],
        pad_size=config["dataset"]["pad_size"],
        metadata_filters={"material": config["dataset"]["material"]},
        conversion_dict_path=config["dataset"]["conversion_dict_path"],
        frac=config["dataset"]["sample_frac"],
        to_normalize=config["dataset"]["to_normalize"],
    )
    print("========================================")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True,
                              num_workers=config["dataset"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False,
                            num_workers=config["dataset"]["num_workers"])

    trainer = pl.Trainer(strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=config["training"]["num_gpus"] if torch.cuda.is_available() else 1,
                         num_nodes=config["training"]["num_nodes"],
                         max_epochs=config["training"]["num_epochs"],
                         gradient_clip_val=200,
                         gradient_clip_algorithm="norm",
                         log_every_n_steps=10,
                         callbacks = ModelCheckpoint(monitor = "val_loss", save_top_k=1, mode = "min")
                         )

    if not os.path.exists(trainer.log_dir):
        os.makedirs(trainer.log_dir, exist_ok=True)
        print(trainer.log_dir)

    shutil.copy(config_path, f"{trainer.log_dir}/config_model.yaml")

    # Extraction des indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # Sauvegarde des indices
    np.save(f"{trainer.log_dir}/train_indices.npy", train_indices)
    np.save(f"{trainer.log_dir}/val_indices.npy", val_indices)

    print("Indices sauvegardés !")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
