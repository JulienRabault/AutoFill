import os
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from SVAE import CustomizableVAE
from VAE_1D import VAE_1D
from conv1DVAE_2_feature import Conv1DVAE_2_feature
from conv1DVAE_concat import Conv1DVAE_concat
from datasetH5 import HDF5Dataset

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
            data_q, data_y, metadata, _ = zip(*batch)
            q = torch.stack(data_q).to("cuda")
            inputs = torch.stack(data_y).to("cuda")
            _, reconstructions, _, _ = model(q, inputs, metadata)

            # Create a single figure with subplots arranged in 2x2 grid
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.ravel()  # Flatten the array for easier iteration

            # Plot each sample in a subplot
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

def main():
    hdf5_file = 'data_pad_90.h5'
    pad_size = 80
    conversion_dict_path = 'conversion_dict.json'
    metadata_filters={"technique": ["les"], "material": ["ag"]}
    # metadata_filters=[]
    dataset = HDF5Dataset(hdf5_file=hdf5_file, pad_size=pad_size, metadata_filters=metadata_filters,
                          conversion_dict_path=conversion_dict_path, frac=0.5)

    # Création du DataLoader
    batch_size = 512
    num_workers = os.cpu_count()
    print(f"Nombre de coeurs disponibles: {num_workers}")

    # Séparation train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
    )

    # Définir les paramètres du VAE
    latent_dim = 32
    learning_rate = 1e-4

    # vae = Conv1DVAE_2_feature(pad_size, latent_dim, learning_rate=learning_rate)
    # vae = Conv1DVAE_concat(pad_size, latent_dim, learning_rate=learning_rate, beta=0.5)
    # vae = VAE_1D(
    #     input_dim=pad_size,
    #     learning_rate=learning_rate,
    #     latent_dim=latent_dim
    #     , beta=1
    # )
    vae = CustomizableVAE(
        in_channels=1,
        out_channels=1,
        down_channels=[32, 64, 128, 256],
        up_channels=[256, 128, 64, 32],
        down_rate=[2, 2, 2, 2],
        up_rate=[2, 2, 2, 2],
        cross_attention_dim=64,
        learning_rate=5e-4,
        beta=0.00001
    )
    # Configuration des Loggers
    logger_tb = TensorBoardLogger("logs_h5", name="vae_model")
    logger_csv = CSVLogger("logs_h5", name="vae_model_csv")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='vae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    inference_plot_callback = InferencePlotCallback(val_loader, output_dir="logs_h5/vae_model")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    # Instanciation du Trainer
    trainer = Trainer(
        max_epochs=50,
        devices='auto',
        logger=[logger_tb, logger_csv],
        callbacks=[checkpoint_callback, early_stopping,inference_plot_callback],
        log_every_n_steps=10,
    )

    # Entraînement
    trainer.fit(vae, train_loader, val_loader)



if __name__ == "__main__":
    main()
