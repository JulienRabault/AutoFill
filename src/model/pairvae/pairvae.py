import torch
from torch import nn

from src.model.vae.pl_vae import PlVAE


class PairVAE(nn.Module):
    """
    Classe PairVAE pour l'entraînement par paire avec reconstruction croisée
    et alignement des espaces latents.
    """

    def __init__(self, config):
        super(PairVAE, self).__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("========================================")
        print("INIT VAE SAXS")

        self.vae_saxs = PlVAE.load_from_checkpoint(self.config["VAE_SAXS"]["path_checkpoint"]).to(self.device)
        print("========================================")

        print("========================================")
        print("INIT VAE LES")
        self.vae_les = PlVAE.load_from_checkpoint(self.config["VAE_SAXS"]["path_checkpoint"]).to(self.device)
        print("========================================")

    def forward(self, batch):
        """
        Réalise l'encodage, la reconstruction et les reconstructions croisées.

        Paramètres:
            batch (dict): Contient :
                - "data_saxs" : images SAXS.
                - "data_les" : images LES.

        Renvoie:
            dict: Dictionnaire contenant les reconstructions et variables latentes.
        """
        metadata = batch["metadata"]

        # Domaine SAXS
        y_saxs = batch["data_y_saxs"]
        q_saxs = batch["data_q_saxs"]
        batch_saxs = {"data_y": y_saxs, "data_q": q_saxs, "metadata": metadata}
        output_saxs = self.vae_saxs(batch_saxs)
        recon_saxs = output_saxs["recon"]
        mu_saxs = output_saxs["mu"]
        logvar_saxs = output_saxs["logvar"]
        z_saxs = output_saxs["z"]

        # Domaine LES
        y_les = batch["data_y_les"]
        q_les = batch["data_q_les"]
        batch_les = {"data_y": y_les, "data_q": q_les, "metadata": metadata}
        output_les = self.vae_les(batch_les)
        recon_les = output_les["recon"]
        mu_les = output_les["mu"]
        logvar_les = output_les["logvar"]
        z_les = output_les["z"]

        # Reconstructions croisées
        recon_les2saxs = self.vae_saxs.decode(z_les)
        recon_saxs2les = self.vae_les.decode(z_saxs)

        return {
            "recon_saxs": recon_saxs,
            "recon_les": recon_les,
            "recon_saxs2les": recon_saxs2les,
            "recon_les2saxs": recon_les2saxs,
            "z_saxs": z_saxs,
            "z_les": z_les
        }
