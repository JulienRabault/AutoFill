import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from VAE_1D import VAE_1D
from conv1DVAE_2_feature import Conv1DVAE_2_feature
from conv1DVAE_concat import Conv1DVAE_concat
from datasetTXT import CustomDatasetVAE


def collate_batch(batch):
    return batch

def main():
    data_csv_path = '../AUTOFILL_data/datav2/merged_cleaned_data.csv'
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Le fichier CSV spécifié est introuvable: {data_csv_path}")
    dataframe = pd.read_csv(data_csv_path)
    dataframe_LES = dataframe[(dataframe['technique'] == 'les')].sample(frac=0.2, random_state=0)
    pad_size = 81  # Ajustez selon vos besoins
    dataset = CustomDatasetVAE(dataframe=dataframe_LES, data_dir='../AUTOFILL_data/datav2/Base_de_donnee',
                               pad_size=pad_size)

    # Création du DataLoader
    batch_size = 512
    num_workers = 14

    # Séparation train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch
    )

    # Définir les paramètres du VAE
    latent_dim = 32
    learning_rate = 1e-4

    hidden_dim = 64
    # vae = Conv1DVAE_2_feature(pad_size, latent_dim, learning_rate=learning_rate)

    vae = VAE_1D(
        input_dim=pad_size,
        learning_rate=learning_rate,
        latent_dim=latent_dim,
    )

    # Configuration des Loggers
    logger_tb = TensorBoardLogger("tb_logs", name="vae_model")
    logger_csv = CSVLogger("csv_logs", name="vae_model")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='vae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    # Instanciation du Trainer
    trainer = Trainer(
        max_epochs=1,
        devices='auto',
        logger=[logger_tb, logger_csv],
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
                        profiler="simple"
    )

    # Entraînement
    trainer.fit(vae, train_loader, val_loader)



if __name__ == "__main__":
    main()
