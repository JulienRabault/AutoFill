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

from SVAE import CustomizableVAE
from VAE_1D import VAE_1D
from conv1DVAE_2_feature import Conv1DVAE_2_feature
from conv1DVAE_concat import Conv1DVAE_concat
from datasetTXT import CustomDatasetVAE
from mainH5 import InferencePlotCallback


def collate_batch(batch):
    return batch


def parse_args():
    """
    Analyse les arguments de la ligne de commande.
    """
    parser = argparse.ArgumentParser(description="Entraînement d'un VAE customisable")
    # Chemins et filtres dataset
    parser.add_argument("--data_csv_path", type=str, default="../AUTOFILL_data/datav2/merged_cleaned_data.csv",
                        help="Chemin vers le fichier CSV des données")
    parser.add_argument("--data_dir", type=str, default="../AUTOFILL_data/datav2/Base_de_donnee",
                        help="Répertoire contenant les données")
    parser.add_argument("--technique", type=str, default="les",
                        help="Filtre pour la colonne 'technique'")
    parser.add_argument("--material", type=str, default="ag",
                        help="Filtre pour la colonne 'material'")
    parser.add_argument("--sample_frac", type=float, default=0.8,
                        help="Fraction d'échantillonnage du dataset après filtrage")
    # Paramètres d'entraînement
    parser.add_argument("--log_dir", type=str, default="logs_txt",
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
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Taille du batch")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="Nombre maximum d'époques")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience pour l'arrêt précoce")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Nombre de workers pour le DataLoader")
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

    def load_dataframe(self):
        """
        Charge le dataframe depuis le fichier CSV.
        """
        if not os.path.exists(self.args.data_csv_path):
            raise FileNotFoundError(f"Fichier CSV introuvable: {self.args.data_csv_path}")
        dataframe = pd.read_csv(self.args.data_csv_path)
        print(f"Nombre total de données: {len(dataframe)}")
        return dataframe

    def filter_dataframe(self, dataframe):
        """
        Applique les filtres sur le dataframe.
        """
        df_filtered = dataframe[dataframe['technique'] == self.args.technique]
        df_filtered = df_filtered[df_filtered['material'] == self.args.material].sample(frac=self.args.sample_frac)
        print(f"Nombre de données après filtrage: {len(df_filtered)}")
        return df_filtered

    def create_dataloaders(self, df_filtered):
        """
        Crée les DataLoaders pour l'entraînement et la validation.
        """
        dataset = CustomDatasetVAE(
            dataframe=df_filtered,
            data_dir=self.args.data_dir,
            pad_size=self.args.pad_size
        )
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch
        )
        print(f"Train size: {train_size}, Validation size: {val_size}")
        return train_loader, val_loader

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
                learning_rate=self.args.learning_rate,
                beta=self.args.beta
            )
        elif self.args.model == "vae1d":
            model = VAE_1D(
                input_dim=self.args.pad_size,
                learning_rate=self.args.learning_rate,
                latent_dim=self.args.latent_dim,
                beta=self.args.beta
            )
        elif self.args.model == "conv1d_2_feature":
            model = Conv1DVAE_2_feature(
                self.args.pad_size,
                self.args.latent_dim,
                learning_rate=self.args.learning_rate
            )
        elif self.args.model == "conv1d_concat":
            model = Conv1DVAE_concat(
                self.args.pad_size,
                self.args.latent_dim,
                learning_rate=self.args.learning_rate
            )
        else:
            raise ValueError("Modèle inconnu.")
        return model

    def create_loggers(self):
        """
        Configure les loggers TensorBoard et CSV.
        """
        logger_tb = TensorBoardLogger(self.args.log_dir, name="vae_model")
        logger_csv = CSVLogger(self.args.log_dir, name="vae_model_csv")
        return [logger_tb, logger_csv]

    def create_callbacks(self, val_loader):
        """
        Crée les callbacks pour le training.
        """
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='vae-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        inference_plot_callback = InferencePlotCallback(val_loader, output_dir=self.args.log_dir)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.args.patience,
            verbose=True,
            mode='min'
        )
        return [checkpoint_callback, early_stopping, inference_plot_callback]

    def run(self):
        """
        Exécute le processus d'entraînement.
        """
        dataframe = self.load_dataframe()
        df_filtered = self.filter_dataframe(dataframe)
        train_loader, val_loader = self.create_dataloaders(df_filtered)
        model = self.create_model()
        loggers = self.create_loggers()
        callbacks = self.create_callbacks(val_loader)
        trainer = Trainer(
            max_epochs=self.args.max_epochs,
            devices="auto",
            logger=loggers,
            callbacks=callbacks,
            log_every_n_steps=10,
        )
        trainer.fit(model, train_loader, val_loader)


def main():
    args = parse_args()
    trainer_manager = TrainingManager(args)
    trainer_manager.run()


if __name__ == "__main__":
    main()
