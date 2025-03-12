import torch
from torch import nn
import torch.nn.functional as F

from model.VAE.submodel.SVAE import CustomizableVAE

class PairVAE(nn.Module):
    """
    Classe PairVAE pour l'entraînement par paire avec reconstruction croisée
    et alignement des espaces latents.
    """
    def __init__(self, config):
        super(PairVAE, self).__init__()

        self.config = config

        self.vae_saxs = CustomizableVAE(**self.config["VAE_SAXS"]["model"])
        if self.config["VAE_SAXS"]["path_checkpoint"] is not None :
            checkpoint = torch.load(self.config["VAE_SAXS"]["path_checkpoint"], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.vae_saxs.load_state_dict(checkpoint['state_dict'])

        self.vae_les = CustomizableVAE(**self.config["VAE_LES"]["model"])
        if self.config["VAE_LES"]["path_checkpoint"] is not None :
            checkpoint = torch.load(self.config["VAE_LES"]["path_checkpoint"], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.vae_les.load_state_dict(checkpoint['state_dict'])

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
        # Domaine SAXS
        x_saxs = batch["data_saxs"]
        mu_saxs, logvar_saxs = self.vae_saxs.encode(x_saxs)
        z_saxs = self.vae_saxs.reparameterize(mu_saxs, logvar_saxs)
        recon_saxs = self.vae_saxs.decode(z_saxs)

        # Domaine LES
        x_les = batch["data_les"]
        mu_les, logvar_les = self.vae_les.encode(x_les)
        z_les = self.vae_les.reparameterize(mu_les, logvar_les)
        recon_les = self.vae_les.decode(z_les)

        # Reconstructions croisées
        # SAXS→LES
        recon_saxs2les = self.vae_saxs.decode(z_les)
        # LES→SAXS
        recon_les2saxs = self.vae_les.decode(z_saxs)

        return {
            "recon_saxs": recon_saxs,
            "recon_les": recon_les,
            "recon_saxs2les": recon_saxs2les,
            "recon_les2saxs": recon_les2saxs,
            "z_saxs": z_saxs,
            "z_les": z_les
        }