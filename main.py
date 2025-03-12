#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from model.VAE.pl_VAE import PlVAE

from dataset.datasetH5 import HDF5Dataset
from model.VAE.submodel.VAE_1D import VAE_1D
from model.VAE.submodel.SVAE import CustomizableVAE
from model.VAE.submodel.conv1DVAE_concat import Conv1DVAE_concat
from model.VAE.submodel.conv1DVAE_2_feature import Conv1DVAE_2_feature

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

            for i in range(min(len(inputs), 4)):
                ax = axs[i]
                ax.plot(inputs[i].cpu().numpy(), label="Original")
                ax.plot(reconstructions[i].cpu().numpy(), label="Reconstructed")
                ax.set_title(f"Sample {i}")
                ax.legend()
                ax.grid(True)

            # Adjust layout and save
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "combined_reconstructions.png")
            plt.savefig(plot_path)
            plt.close()


def collate_batch(batch):
    return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un VAE customisable")
    # Chemins et filtres dataset
    parser.add_argument("--name", type=str, default="customizable_vae",)
    parser.add_argument("--devices", type=int,)
    parser.add_argument("--hdf5_file", type=str, default="data.h5",
                        help="Chemin vers le fichier HDF5 des données")
    parser.add_argument("--conversion_dict_path", type=str, default="conversion_dict.json",
                        help="Chemin vers le dictionnaire de conversion des métadonnées")
    parser.add_argument("--technique", type=str, default="les",
                        help="Filtre pour la colonne 'technique'")
    parser.add_argument("--material", type=str, default="ag",
                        help="Filtre pour la colonne 'material'")
    # Paramètres d'entraînement
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="Répertoire de sortie pour les logs")
    parser.add_argument("--model", type=str, default="customizable",
                        choices=["customizable", "vae1d", "conv1d_2_feature", "conv1d_concat"],
                        help="Modèle à utiliser")
    parser.add_argument("--pad_size", type=int, default=80,
                        help="Taille de padding des séquences")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Dimension latente")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Taux d'apprentissage. Par défaut: 5e-4 pour 'customizable', 1e-4 sinon")
    parser.add_argument("--beta", type=float, default=0.00001,
                        help="Coefficient beta pour le VAE")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Taille du batch")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Nombre maximum d'époques")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience pour l'arrêt précoce")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Nombre de workers pour le DataLoader")
    parser.add_argument("--sample_frac", type=float, default=1,
                        help="Fraction d'échantillonnage du dataset après filtrage")
    return parser.parse_args()


class TrainingManager:
    """
    Classe pour gérer le processus d'entraînement.
    """

    def __init__(self, args):
        """
        Initialisation du gestionnaire d'entraînement.
        """
        self.args = args
        # Définir le learning rate par défaut selon le modèle
        if self.args.learning_rate is None:
            if self.args.model == "customizable":
                self.args.learning_rate = 5e-4
            else:
                self.args.learning_rate = 1e-4


    def load_dataset(self):
        dataset = HDF5Dataset(
            hdf5_file=self.args.hdf5_file,
            pad_size=self.args.pad_size,
            metadata_filters={"technique": [self.args.technique], "material": [self.args.material]},
            conversion_dict_path=self.args.conversion_dict_path,
            frac=self.args.sample_frac

        )
        return dataset

    def create_model(self):
        """
        Instancie le modèle selon le choix utilisateur.
        """
        if self.args.model == "customizable":
            model = CustomizableVAE(
                in_channels=1,
                out_channels=1,
                down_channels=[32, 64, 128, 256],
                up_channels=[256, 128, 64, 32],
                down_rate=[2, 2, 2, 2],
                up_rate=[2, 2, 2, 2],
                cross_attention_dim=64,
            )

        elif self.args.model == "vae1d":
            model = VAE_1D(
                input_dim=self.args.pad_size,
                latent_dim=self.args.latent_dim,
            )

        elif self.args.model == "conv1d_2_feature":
            model = Conv1DVAE_2_feature(
                self.args.pad_size,
                self.args.latent_dim,
            )

        elif self.args.model == "conv1d_concat":
            model = Conv1DVAE_concat(
                self.args.pad_size,
                self.args.latent_dim,
            )

        else:
            raise ValueError("Modèle inconnu.")
        
        return PlVAE(model, learning_rate=self.args.learning_rate, beta = self.args.beta)

    def create_loggers(self):
        """
        Configure les loggers TensorBoard et CSV.
        """
        logger_tb = TensorBoardLogger(self.args.log_dir, name=self.args.name)
        logger_csv = CSVLogger(self.args.log_dir, name=self.args.name)
        return [logger_tb, logger_csv]

    def create_callbacks(self, val_loader):
        """
        Crée les callbacks pour le training.
        """
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='vae-{epoch:02d}-{val_loss:.5f}',
            save_top_k=1,
            mode='min',
        )
        inference_plot_callback = InferencePlotCallback(val_loader, output_dir=os.path.join(self.args.log_dir, self.args.name))
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.args.patience,
            verbose=True,
            mode='min'
        )
        return [checkpoint_callback, early_stopping, inference_plot_callback]

    def run(self):
        dataset = self.load_dataset()
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        model = self.create_model()
        loggers = self.create_loggers()
        callbacks = self.create_callbacks(val_loader)
        trainer = Trainer(max_epochs=self.args.max_epochs, logger=loggers, callbacks=callbacks, log_every_n_steps=10, accelerator="gpu", devices=self.args.devices)
        trainer.fit(model, train_loader, val_loader)


def main():
    args = parse_args()
    trainer_manager = TrainingManager(args)
    trainer_manager.run()


if __name__ == "__main__":
    main()