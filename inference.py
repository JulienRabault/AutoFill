import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

from SVAE import CustomizableVAE
from datasetH5 import HDF5Dataset

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

def calculate_metrics(original, reconstructed, data_range=0.1):
    """
    Calcule les métriques pour évaluer la qualité de la reconstruction.
    """
    mse = np.mean((original - reconstructed) ** 2)
    psnr_value = psnr(original, reconstructed, data_range=data_range)
    ssim_value = ssim(original, reconstructed, data_range=data_range)

    return mse, psnr_value, ssim_value


def evaluate_model(model, dataloader, output_dir="inference_results"):
    """
    Évalue le modèle sur les données de test et calcule les métriques.
    """
    model.eval()
    mse_values = []
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = transform_batch(batch)
            _, reconstructions, _, _ = model(batch)
            inputs = batch["data_y"].cpu().numpy()
            reconstructions = reconstructions.cpu().numpy()

            for i in range(len(inputs)):
                mse, psnr_value, ssim_value = calculate_metrics(inputs[i], reconstructions[i])
                mse_values.append(mse)
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)

    avg_mse = np.mean(mse_values)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"MSE moyen: {avg_mse}") # le plus petit est le mieux
    print(f"PSNR moyen: {avg_psnr}") # le plus grand est le mieux
    print(f"SSIM moyen: {avg_ssim}") # le plus proche de 1 est le mieux

    metrics = pd.DataFrame({
        "MSE": mse_values,
        "PSNR": psnr_values,
        "SSIM": ssim_values
    })

    metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    return avg_mse, avg_psnr, avg_ssim

def collate_batch(batch):
    return batch

def load_model(checkpoint_path, input_dim, latent_dim, learning_rate):
    """
    Charge un modèle VAE pré-entraîné à partir d'un fichier checkpoint.
    """
    model = CustomizableVAE(
        in_channels=1,
        input_dim=self.args.pad_size,
        latent_dim=32,
        learning_rate=self.args.learning_rate,
        beta=self.args.beta
    )

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model.to("cuda")

def transform_batch(batch):
    data_q_list = []
    data_y_list = []
    metadata_list = []

    for item in batch:
        data_q_list.append(item["data_q"])
        data_y_list.append(item["data_y"])
        metadata_list.append(item["metadata"])

    data_q_tensor = torch.stack(data_q_list)
    data_y_tensor = torch.stack(data_y_list)
    metadata_tensor = torch.stack(metadata_list)

    transformed_batch = {
        "data_q": data_q_tensor.to("cuda"),
        "data_y": data_y_tensor.to("cuda"),
        "metadata": metadata_tensor.to("cuda")
    }

    return transformed_batch

def infer_and_plot(model, dataloader, output_dir="inference_results", num_samples=5):
    """
    Effectue l'inférence et enregistre les résultats de reconstruction.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch = transform_batch(batch)
        for el in batch:
            if isinstance(batch[el], torch.Tensor):
                batch[el] = batch[el].to("cuda")

        _, reconstructions, _, _ = model(batch)
        inputs = batch["data_y"]
        for i in range(min(len(inputs), num_samples)):
            plt.figure(figsize=(10, 4))
            plt.plot(inputs[i].cpu().numpy(), label="Original")
            plt.plot(reconstructions[i].cpu().numpy(), label="Reconstructed")
            plt.title(f"Sample {len(inputs) + i}")
            plt.legend()
            plt.grid(True)

            plot_path = os.path.join(output_dir, f"sample_{len(inputs) + i}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Reconstruction enregistrée dans : {plot_path}")

def plot_training_curves(csv_log_dir, output_dir="inference_results"):
    # Lire le fichier CSV
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(csv_log_dir, "metrics.csv")

    if not os.path.exists(log_file):
        print(f"Aucun fichier de log trouvé à {log_file}")
        return

    data = pd.read_csv(log_file)

    loss_columns = [
        'kl_loss_epoch', 'kl_loss_step',
        'recon_loss_epoch', 'recon_loss_step',
        'train_loss_epoch', 'train_loss_step',
        'val_loss'
    ]

    # Créer une figure pour les époques
    fig_epoch, axs_epoch = plt.subplots(len(loss_columns) // 2, 1, figsize=(10, 12))
    fig_epoch.suptitle('Loss vs Epoch')

    # Créer une figure pour les étapes
    fig_step, axs_step = plt.subplots(len(loss_columns) // 2, 1, figsize=(10, 12))
    fig_step.suptitle('Loss vs Step')

    epoch_index = 0
    step_index = 0

    # Générer les courbes
    for loss_col in loss_columns:
        if 'epoch' in loss_col:
            x_col = 'epoch'
            filtered_df = data.dropna(subset=[loss_col, x_col])
            axs_epoch[epoch_index].plot(filtered_df[x_col], filtered_df[loss_col], label=loss_col)
            axs_epoch[epoch_index].set_xlabel('epoch')
            axs_epoch[epoch_index].set_ylabel(loss_col)
            axs_epoch[epoch_index].set_title(f'{loss_col} vs {x_col}')
            axs_epoch[epoch_index].legend()
            axs_epoch[epoch_index].grid(True)
            epoch_index += 1
        elif 'step' in loss_col:
            x_col = 'step'
            filtered_df = data.dropna(subset=[loss_col, x_col])
            axs_step[step_index].plot(filtered_df[x_col], filtered_df[loss_col], label=loss_col)
            axs_step[step_index].set_xlabel('step')
            axs_step[step_index].set_ylabel(loss_col)
            axs_step[step_index].set_title(f'{loss_col} vs {x_col}')
            axs_step[step_index].legend()
            axs_step[step_index].grid(True)
            step_index += 1

    # Sauvegarder les figures
    plt.tight_layout()
    fig_epoch.savefig(os.path.join(output_dir, 'loss_vs_epoch.png'))
    fig_step.savefig(os.path.join(output_dir, 'loss_vs_step.png'))
    plt.close(fig_epoch)
    plt.close(fig_step)

def main(args):
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {args.checkpoint_path}")

    dataset_test = HDF5Dataset(
        hdf5_file=args.hdf5_file,
        pad_size=args.pad_size,
        metadata_filters={"technique": [args.technique], "material": [args.material]},
        conversion_dict_path=args.conversion_dict_path,
        frac=args.frac
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    model = load_model(args.checkpoint_path, input_dim=args.input_dim, latent_dim=args.latent_dim, learning_rate=args.learning_rate)

    infer_and_plot(model, test_loader, output_dir=args.output_dir, num_samples=args.num_samples)

    plot_training_curves(csv_log_dir=args.csv_log_dir)

    evaluate_model(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script pour l'inférence et la visualisation des résultats d'un modèle VAE.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Chemin vers le fichier checkpoint du modèle.")
    parser.add_argument("--csv_log_dir", type=str, required=True, help="Répertoire contenant les fichiers de log CSV.")
    parser.add_argument("--technique", type=str, default="les",
                        help="Filtre pour la colonne 'technique'")
    parser.add_argument("--material", type=str, default="ag",
                        help="Filtre pour la colonne 'material'")
    parser.add_argument("--hdf5_file", type=str, default="data.h5", help="Chemin vers le fichier HDF5 contenant les données.")
    parser.add_argument("--conversion_dict_path", type=str, default="conversion_dict.json", help="Chemin vers le fichier JSON de conversion.")
    parser.add_argument("--pad_size", type=int, default=80, help="Taille de padding pour les données.")
    parser.add_argument("--frac", type=float, default=1, help="Fraction des données à utiliser.")
    parser.add_argument("--batch_size", type=int, default=16, help="Taille du batch pour le DataLoader.")
    parser.add_argument("--input_dim", type=int, default=80, help="Dimension d'entrée du modèle.")
    parser.add_argument("--latent_dim", type=int, default=32, help="Dimension latente du modèle.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Taux d'apprentissage du modèle.")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Répertoire de sortie pour les résultats d'inférence.")
    parser.add_argument("--num_samples", type=int, default=5, help="Nombre d'échantillons à tracer.")

    args = parser.parse_args()
    main(args)
