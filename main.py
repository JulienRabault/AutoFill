#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from model.VAE.pl_VAE import PlVAE

from dataset.datasetH5 import HDF5Dataset

class InferencePlotCallback(pl.Callback):
    def __init__(self, dataloader, output_dir="inference_results"):
        super().__init__()
        self.dataloader = dataloader
        self.output_dir = output_dir

    def on_validation_end(self, trainer, pl_module):
        self.infer_and_plot(pl_module)

    def infer_and_plot(self, model,):
        """
        Perform inference and save reconstruction results in a single plot with subplots.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        model.eval().to("cuda")
        with torch.no_grad():
            batch = next(iter(self.dataloader))
            for el in batch:
                if isinstance(batch[el], torch.Tensor):
                    batch[el] = batch[el].to("cuda")
            data_y = batch["data_y"]
            inputs = data_y

            _, reconstructions, _, _ = model(batch)
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.ravel()

            for i in range(min(len(inputs), 4)):
                ax = axs[i]
                ax.plot(inputs[i].cpu().numpy(), label="Original")
                ax.plot(reconstructions[i].cpu().numpy(), label="Reconstructed")
                ax.set_title(f"Sample {i}")
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "combined_reconstructions.png")
            plt.savefig(plot_path)
            plt.close()


def collate_batch(batch):
    return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un VAE customisable")
    # Chemins et filtres dataset
    parser.add_argument("--config", type=str, default="model/VAE/vae_config.yaml",)
    parser.add_argument("--name", type=str, default="customizable_vae", )
    parser.add_argument("--conversion_dict_path", type=str, default="conversion_dict_saxs.json",
                        help="Chemin vers le dictionnaire de conversion des métadonnées")
    parser.add_argument("--technique", type=str, default="saxs",
                        help="Filtre pour la colonne 'technique'")
    parser.add_argument("--material", type=str, default="ag",
                        help="Filtre pour la colonne 'material'")
    parser.add_argument("--devices")
    # Paramètres d'entraînement
    return parser.parse_args()


class TrainingManager:
    """
    Classe pour gérer le processus d'entraînement.
    """

    def __init__(self, config):
        """
        Initialisation du gestionnaire d'entraînement.
        """
        self.config = config

    def load_dataset(self):
        dataset = HDF5Dataset(
            hdf5_file=self.config['dataset']['hdf5_file'],
            pad_size=self.config['dataset']['pad_size'],
            metadata_filters={"technique": [self.config['dataset']['technique']], "material": [self.config['dataset']['material']]},
            conversion_dict_path=self.config['dataset']['conversion_dict_path'],
            frac=self.config['dataset']['sample_frac'],
            to_normalize=['data_y']
        )
        return dataset

    def create_model(self):
        """
        Instancie le modèle selon le choix utilisateur.
        """
        if self.config['model']['vae_class'] == "CustomizableVAE":
            model = PlVAE(config=self.config['model'])
        else:
            raise ValueError("Modèle inconnu.")

        return model

    def create_loggers(self):
        """
        Configure les loggers TensorBoard et CSV.
        """
        logger_tb = TensorBoardLogger(self.config['log_dir'], name=self.config['name'])
        logger_csv = CSVLogger(self.config['log_dir'], name=self.config['name'])
        return [logger_tb, logger_csv]

    def create_callbacks(self, val_loader):
        """
        Crée les callbacks pour le training.
        """
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='best',
            save_top_k=1,
            mode='min',
        )
        inference_plot_callback = InferencePlotCallback(val_loader, output_dir=os.path.join(self.config['log_dir'],
                                                                                            self.config['name']))
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            verbose=True,
            mode='min'
        )
        return [checkpoint_callback, early_stopping, inference_plot_callback]

    def run(self):
        dataset = self.load_dataset()
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                  num_workers=os.cpu_count())
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=os.cpu_count())
        model = self.create_model()
        loggers = self.create_loggers()
        callbacks = self.create_callbacks(val_loader)
        trainer = Trainer(max_epochs=self.config['max_epochs'], logger=loggers, callbacks=callbacks,
                          log_every_n_steps=10, accelerator="gpu", devices=self.config['devices'])
        trainer.fit(model, train_loader, val_loader)


def main():
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config['name'] = args.name
    config['dataset']['material'] = args.material
    config['dataset']['technique'] = args.technique
    config['devices'] = args.devices
    trainer_manager = TrainingManager(config)
    trainer_manager.run()


if __name__ == "__main__":
    main()