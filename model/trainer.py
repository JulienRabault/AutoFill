import os
import shutil
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from model.pl_PairVAE import PlPairVAE 
from dataset.datasetPairH5 import PairHDF5Dataset

def train(config) : 
    print("========================================")
    print("INIT Model")
    model = PlPairVAE(config)
    print("========================================")
    
    print("========================================")
    print("INIT Dataset")

    dataset = PairHDF5Dataset(
        hdf5_file = config["dataset"]["hdf5_file"],
        metadata_filters = config["dataset"]["metadata_filters"],
        conversion_dict_path = config["dataset"]["conversion_dict_path"],
        sample_frac = config["dataset"]["sample_frac"],
        transform =  config["dataset"]["transform"],
        requested_metadata =  config["dataset"]["requested_metadata"],
    )
    print("========================================")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config["training"]['patience'],
        verbose=True,
        mode='min')
    model_ckpt = ModelCheckpoint(monitor = "val_loss", save_top_k=1, mode = "min")

    trainer = pl.Trainer(strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices=config["training"]["num_gpus"] if torch.cuda.is_available() else 1,
                        num_nodes=config["training"]["num_nodes"],
                        max_epochs=config["training"]["num_epochs"],
                        gradient_clip_val=200,
                        gradient_clip_algorithm="norm",
                        profiler="advanced",
                        log_every_n_steps=10,
                        callbacks = [model_ckpt, early_stopping],
                        )
    
    if not os.path.exists(trainer.log_dir):
        os.makedirs(trainer.log_dir, exist_ok=True)
        print(trainer.log_dir)
        
    # Sauvegarder le fichier YAML
    file_path = f"{trainer.log_dir}/config_model.yaml"
    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    print(f"Fichier YAML sauvegardé dans : {file_path}")

    # Extraction des indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    # Sauvegarde des indices
    np.save(f"{trainer.log_dir}/train_indices.npy", train_indices)
    np.save(f"{trainer.log_dir}/val_indices.npy", val_indices)
    
    print("Indices sauvegardés !")
  
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)    
