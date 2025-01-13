import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from VAE_1D import VAE_1D
from datasetTXT import CustomDatasetVAE


def collate_batch(batch):
    return batch


def load_model(checkpoint_path, input_dim, latent_dim, learning_rate):
    """
    Charge un modèle VAE pré-entraîné à partir d'un fichier checkpoint.
    """
    model = VAE_1D(
        input_dim=input_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate
    )
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
        data_q, data_y, metadata = zip(*batch)  # Décomposer les tuples
        inputs = torch.stack(data_q)  # Empile les données en un tenseur [batch_size, sequence_length]
        _, reconstructions, _, _ = model(inputs)

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
    """
    Trace les courbes d'entraînement et de validation dans des graphes distincts à partir des logs CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(csv_log_dir, "metrics.csv")

    if not os.path.exists(log_file):
        print(f"Aucun fichier de log trouvé à {log_file}")
        return

    data = pd.read_csv(log_file)

    metrics = [
        ("train_loss_step", "Train Loss (Step)"),
        ("val_loss", "Validation Loss"),
        ("kl_loss_step", "KL Loss (Step)"),
        ("recon_loss_step", "Reconstruction Loss (Step)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Aplatir l'array pour accéder facilement aux sous-graphes

    for idx, (column, label) in enumerate(metrics):
        ax = axes[idx]
        if column in data.columns:
            ax.plot(data["step"], data[column], label=label)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend()
        ax.grid(True)

    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.tight_layout()  # Pour éviter que les titres ne se chevauchent
    plt.savefig(plot_path)
    plt.show()
    print(f"Courbes enregistrées dans : {plot_path}")



def main():
    # Chemin du checkpoint et du CSV des données de test
    checkpoint_path = "checkpoints/vae-epoch=06-val_loss=0.01.ckpt"
    data_csv_path = '../AUTOFILL_data/datav2/merged_cleaned_data.csv'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Le fichier CSV spécifié est introuvable: {data_csv_path}")

    # Chargement des données de test
    dataframe = pd.read_csv(data_csv_path)
    dataframe_test = dataframe[(dataframe['technique'] == 'les') & (dataframe['material'] == 'au')].sample(frac=0.2)
    pad_size = 128  # Ajustez selon vos besoins
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
    plot_training_curves(csv_log_dir="csv_logs/vae_model/version_0")


if __name__ == "__main__":
    main()
