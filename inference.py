import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from VAE_1D import VAE_1D
from conv1DVAE_2_feature import Conv1DVAE_2_feature
from conv1DVAE_concat import Conv1DVAE_concat
from datasetTXT import CustomDatasetVAE


def collate_batch(batch):
    return batch


def load_model(checkpoint_path, input_dim, latent_dim, learning_rate):
    """
    Charge un modèle VAE pré-entraîné à partir d'un fichier checkpoint.
    """

    # model = Conv1DVAE_concat(input_dim, latent_dim, learning_rate=learning_rate)

    model = Conv1DVAE_2_feature(input_dim, latent_dim, learning_rate=learning_rate)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def infer_and_plot(model, dataloader, output_dir="inference_results", num_samples=5):
    """
    Effectue l'inférence et enregistre les résultats de reconstruction.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        data_q, data_y, metadata, _ = zip(*batch)  # Décomposer les tuples
        q = torch.stack(data_q)  # Décomposer les tuples
        inputs = torch.stack(data_y)
        _, reconstructions, _, _ = model(q, inputs, metadata)

        # Tracer les résultats pour un nombre limité d'échantillons
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

def plot_training_curves(csv_log_dir, output_dir="plots"):
    # Lire le fichier CSV
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(csv_log_dir, "metrics.csv")

    if not os.path.exists(log_file):
        print(f"Aucun fichier de log trouvé à {log_file}")
        return

    data = pd.read_csv(log_file)

    # Liste des colonnes de perte
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


def main():
    # Chemin du checkpoint et du CSV des données de test
    checkpoint_path = "tb_logs/vae_model/version_10/checkpoints/vae-epoch=06-val_loss=0.26.ckpt"
    data_csv_path = '../AUTOFILL_data/datav2/merged_cleaned_data.csv'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Le fichier CSV spécifié est introuvable: {data_csv_path}")

    # Chargement des données de test
    dataframe = pd.read_csv(data_csv_path)
    dataframe_test = dataframe[(dataframe['technique'] == 'les')].sample(frac=0.1,random_state=0)
    pad_size = 81  # Ajustez selon vos besoins
    dataset_test = CustomDatasetVAE(dataframe=dataframe_test, data_dir='../AUTOFILL_data/datav2/Base_de_donnee',
                                    pad_size=pad_size)

    test_loader = DataLoader(
        dataset_test,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_batch
    )

    # Paramètres du modèle
    latent_dim = 32
    learning_rate = 1e-4

    # Charger le modèle
    model = load_model(checkpoint_path, input_dim=pad_size, latent_dim=latent_dim, learning_rate=learning_rate)

    # Effectuer l'inférence et sauvegarder les résultats
    infer_and_plot(model, test_loader, output_dir="inference_results", num_samples=5)

    # Tracer les courbes d'entraînement
    plot_training_curves(csv_log_dir="csv_logs/vae_model/version_10/")


if __name__ == "__main__":
    main()
