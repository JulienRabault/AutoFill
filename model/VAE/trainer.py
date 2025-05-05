import os

import mlflow
import yaml

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from model.VAE.pl_VAE import PlVAE
from dataset.datasetH5 import HDF5Dataset
from model.VAE.utils.inference_callback import InferencePlotCallback
from model.VAE.utils.metrics_callback import MAEMetricCallback


def train(config):
    
    print("========================================")
    print("INIT Model")
    model = PlVAE(config)
    
    print("========================================")
    print("INIT Dataset")
    dataset = HDF5Dataset(
        hdf5_file=config["dataset"]["hdf5_file"],
        metadata_filters=config["dataset"]["metadata_filters"],
        conversion_dict_path=config["dataset"]["conversion_dict_path"],
        sample_frac=config["dataset"]["sample_frac"],
        transform=config["dataset"]["transform"],
        requested_metadata=config["dataset"]["requested_metadata"],
    )
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=config["training"]["num_workers"]
    )

    print("========================================")

    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config["training"]['patience'],
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        every_n_epochs=5,
    )

    mlflow_logdir = config["logdir"]
    mlflow_logger = MLFlowLogger(
        experiment_name = "AUTOFILL", 
        run_name=config["experiment_name"],
        log_model = True,
        tracking_uri = f"file:{mlflow_logdir}/mlrun",
    )
    mlflow_logger.log_hyperparams(config)

    use_loglog = config["training"]["use_loglog"]
    inference_callback = InferencePlotCallback(val_loader, artifact_file = "val_plot.png", use_loglog=use_loglog)
    train_inference_callback = InferencePlotCallback(train_loader, artifact_file = "train_plot.png", use_loglog=use_loglog)
    
    mae_callback = MAEMetricCallback(val_loader)
    
    trainer = pl.Trainer(
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["training"]["num_gpus"] if torch.cuda.is_available() else 1,
        num_nodes=config["training"]["num_nodes"],
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback, inference_callback, train_inference_callback, mae_callback],
        logger=mlflow_logger,
    )

    log_dir = os.path.join(mlflow_logger.save_dir, mlflow_logger.experiment_id, mlflow_logger.version)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "config_model.yaml")
    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    print(f"Fichier YAML sauvegardé dans : {file_path}")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    #mlflow.pytorch.log_model(model, "model")
    
    print("Fin du train")
