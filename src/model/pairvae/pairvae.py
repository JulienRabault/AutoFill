import torch
import yaml
from torch import nn

from model.VAE.submodel.registry import *


class PairVAE(nn.Module):
    """
    Classe PairVAE pour l'entraînement par paire avec reconstruction croisée
    et alignement des espaces latents.
    """

    def __init__(self, config, load_weights_VAE):
        super(PairVAE, self).__init__()

        self.config = config

        print("========================================")
        print("INIT VAE SAXS")
        if self.config["VAE_SAXS"]["path_config"] is not None:
            with open(self.config["VAE_SAXS"]["path_config"], 'r') as file:
                config_saxs = yaml.safe_load(file)["model"]["args"]
        else:
            config_saxs = self.config["VAE_SAXS"]["args"]
        print(config_saxs)
        vae_saxs_class = self.config["VAE_SAXS"]["vae_class"]
        self.vae_saxs = MODEL_REGISTRY.get(vae_saxs_class)(**config_saxs)
        if self.config["VAE_SAXS"]["path_checkpoint"] is not None and load_weights_VAE:
            print("LOADING CKPT SAXS")
            checkpoint = torch.load(self.config["VAE_SAXS"]["path_checkpoint"],
                                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            checkpoint['state_dict'] = {k[len("model."):]: v for k, v in checkpoint['state_dict'].items()}
            self.vae_saxs.load_state_dict(checkpoint['state_dict'])
        print("========================================")

        print("========================================")
        print("INIT VAE LES")
        if self.config["VAE_LES"]["path_config"] is not None:
            with open(self.config["VAE_LES"]["path_config"], 'r') as file:
                config_les = yaml.safe_load(file)["model"]["args"]
        else:
            config_les = self.config["VAE_LES"]["args"]
        print(config_les)
        vae_les_class = self.config["VAE_LES"]["vae_class"]
        self.vae_les = MODEL_REGISTRY.get(vae_les_class)(**config_les)
        if self.config["VAE_LES"]["path_checkpoint"] is not None and load_weights_VAE:
            print("LOADING CKPT LES")
            checkpoint = torch.load(self.config["VAE_LES"]["path_checkpoint"],
                                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            checkpoint['state_dict'] = {k[len("model."):]: v for k, v in checkpoint['state_dict'].items()}
            self.vae_les.load_state_dict(checkpoint['state_dict'])
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
        output_saxs = self.vae_saxs(y=y_saxs, q=q_saxs, metadata=metadata)
        recon_saxs = output_saxs["recon"]
        mu_saxs = output_saxs["mu"]
        logvar_saxs = output_saxs["logvar"]
        z_saxs = output_saxs["z"]

        # Domaine LES
        y_les = batch["data_y_les"]
        q_les = batch["data_q_les"]
        output_les = self.vae_les(y=y_les, q=q_les, metadata=metadata)
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
