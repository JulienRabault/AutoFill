import argparse
import itertools
import copy
import os
import yaml
import torch
import mlflow
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from model.VAE.pl_VAE import PlVAE
from dataset.datasetH5 import HDF5Dataset
from model.inference_callback import InferencePlotCallback
from model.metrics_callback import MAEMetricCallback

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
    print("========================================")
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
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config["training"]['patience'],
        verbose=True,
        mode='min'
    )
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join("runs", config["experiment_name"]), monitor="val_loss", save_top_k=1, mode="min"
    )
    mlflow_logger = MLFlowLogger(
        experiment_name="AUTOFILL", run_name=config["experiment_name"],
        tracking_uri="file:runs/mlrun",
    )
    mlflow_logger.log_hyperparams(config)
    use_loglog = config["training"]["use_loglog"]
    inference_callback = InferencePlotCallback(train_loader, output_dir=os.path.join("runs", config["experiment_name"]), use_loglog=use_loglog)
    mae_callback = MAEMetricCallback(val_loader)
    trainer = pl.Trainer(
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["training"]["num_gpus"] if torch.cuda.is_available() else 1,
        num_nodes=config["training"]["num_nodes"],
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=15,
        callbacks=[model_ckpt, early_stopping, inference_callback, mae_callback],
        logger=mlflow_logger,
    )
    log_dir = os.path.join("runs", config["experiment_name"])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(log_dir)
    file_path = os.path.join(log_dir, "config_model.yaml")
    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    print(f"Fichier YAML sauvegardé dans : {file_path}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Fin du train")

def grid_search(base_config):
    if base_config["dataset"]["metadata_filters"]["technique"][0] == "les":
        betas = [0.0001, 0.00001, 0.000001]
        batch_sizes = [32, 64, 128, 256]
        latent_dims = [64, 128, 256]
    else:
        betas = [0.001, 0.0001, 0.00001, 0.000001]
        latent_dims = [64, 128, 256, 512]
        batch_sizes = [32, 64, 128]

    for beta, latent_dim, batch_size in itertools.product(
        betas, latent_dims, batch_sizes
    ):
        config = copy.deepcopy(base_config)
        mat = config["dataset"]["metadata_filters"]["material"]
        tech = config["dataset"]["metadata_filters"]["technique"]
        config["training"]["beta"] = beta
        config["training"]["batch_size"] = batch_size
        config["model"]["args"]["latent_dim"] = latent_dim
        config["experiment_name"] = (
            f"grid_{mat[0]}_{tech[0]}_beta{beta}_ld{latent_dim}_bs{batch_size}"
        )
        print(f"Running experiment: {config['experiment_name']}")
        print("========================================")
        train(config)
        print("========================================")
        print("End of Experiment")
        print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        base_config = yaml.safe_load(file)
    grid_search(base_config)
