import argparse
import itertools
import copy
import os
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.loggers import MLFlowLogger

from model.pl_PairVAE import PlPairVAE 
from dataset.datasetPairH5 import PairHDF5Dataset
from model.utils.inference_callback import PairInferencePlotCallback


def train(config):
    
    print("========================================")
    print("INIT Model")
    model = PlPairVAE(config)

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
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])
    print("========================================")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config["training"]['patience'],
        verbose=True,
        mode='min'
    )

    os.environ["MLFLOW_TRACKING_URI"] = "https://mlflowts.irit.fr"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PNRIA208"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "p^Te85E7b#"

    mlflow_logger = MLFlowLogger(
        experiment_name = "AUTOFILL", 
        run_name=config["experiment_name"],
        tracking_uri = "https://mlflowts.irit.fr",
    )
    mlflow_logger.log_hyperparams(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        every_n_epochs=1,
        dirpath= os.path.join("runs_pair", config["experiment_name"]),
        filename="best",
    )

    use_loglog = config["training"]["use_loglog"]
    inference_callback = PairInferencePlotCallback(val_loader, artifact_file = "val_plot.png", output_dir=os.path.join("runs_pair", config["experiment_name"]), use_loglog=use_loglog)
    train_inference_callback = PairInferencePlotCallback(train_loader, artifact_file = "train_plot.png", output_dir=os.path.join("runs_pair", config["experiment_name"]), use_loglog=use_loglog)
        
    trainer = pl.Trainer(
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["training"]["num_gpus"] if torch.cuda.is_available() else 1,
        num_nodes=config["training"]["num_nodes"],
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback, inference_callback, train_inference_callback],
        logger=mlflow_logger,
    )

    try:
        log_dir = os.path.join(mlflow_logger.save_dir, mlflow_logger.experiment_id, mlflow_logger.version)
    except:
        log_dir = os.path.join("runs_pair", config["experiment_name"])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "config_model.yaml")
    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    print(f"Fichier YAML sauvegardé dans : {file_path}")
    trainer.logger.experiment.log_artifact(local_path=file_path, run_id=trainer.logger.run_id)


    # Extraction des indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    # Sauvegarde des indices
    np.save(f"{log_dir}/train_indices.npy", train_indices)
    np.save(f"{log_dir}/val_indices.npy", val_indices)
    # save artifact mlflow 
    trainer.logger.experiment.log_artifact(local_path=f"{log_dir}/train_indices.npy", run_id=trainer.logger.run_id)
    trainer.logger.experiment.log_artifact(local_path=f"{log_dir}/val_indices.npy", run_id=trainer.logger.run_id)
    
    print("Indices sauvegardés !")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    #mlflow.pytorch.log_model(model, "model")

def grid_search(base_config):
    weight_latent_similarities = [0.0001, 0.00001]
    weight_saxs2saxs_list = [0.04, 0.08, 0.12]
    weight_saxs2les_list = [0.5, 0.05, 0.005]
    weight_les2les_list = [0.04, 0.12]
    weight_les2saxs_list = [0.05, 0.005]

    for weight_latent_similarity, weight_saxs2saxs, weight_saxs2les, weight_les2les, weight_les2saxs in itertools.product(
        weight_latent_similarities, weight_saxs2saxs_list,
        weight_saxs2les_list, weight_les2les_list, weight_les2saxs_list
    ):
        config = copy.deepcopy(base_config)
        mat = config["dataset"]["metadata_filters"]["material"]
        config["training"]["weight_latent_similarity"] = weight_latent_similarity
        config["training"]["weight_saxs2saxs"] = weight_saxs2saxs
        config["training"]["weight_saxs2les"] = weight_saxs2les
        config["training"]["weight_les2les"] = weight_les2les
        config["training"]["weight_les2saxs"] = weight_les2saxs
        config["experiment_name"] = (
            f"grid_{mat[0]}_wlat{weight_latent_similarity}_wsaxs2saxs{weight_saxs2saxs}"
            f"_wsaxs2les{weight_saxs2les}_wles2les{weight_les2les}_wles2saxs{weight_les2saxs}"
        )
        train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        base_config = yaml.safe_load(file)
    grid_search(base_config)
