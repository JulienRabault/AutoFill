import yaml

import torch
from torch import nn
import torch.nn.functional as F

from model.VAE.submodel.registry import *

class PairVAE(nn.Module):
    """
    Classe PairVAE pour l'entraînement par paire avec reconstruction croisée
    et alignement des espaces latents.
    """
    def __init__(self, config):
        super(PairVAE, self).__init__()

        self.config = config


        print("========================================")
        print("INIT VAE SAXS")
        if self.config["VAE_SAXS"]["path_config"] is not None :
            with open(self.config["VAE_SAXS"]["path_config"], 'r') as file:
                config_saxs = yaml.safe_load(file)
        else :
            config_saxs = self.config["VAE_SAXS"]["args"]
        print(config_saxs)
        vae_saxs_class =  self.config["VAE_SAXS"]["vae_class"]
        self.vae_saxs = MODEL_REGISTRY.get(vae_saxs_class)(**config_saxs)
        if self.config["VAE_SAXS"]["path_checkpoint"] is not None :
            checkpoint = torch.load(self.config["VAE_SAXS"]["path_checkpoint"], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.vae_saxs.load_state_dict(checkpoint['state_dict'])
        print("========================================")

        
        print("========================================")
        print("INIT VAE LES")
        if self.config["VAE_LES"]["path_config"] is not None :
            with open(self.config["VAE_LES"]["path_config"], 'r') as file:
                config_les = yaml.safe_load(file)
        else :
            config_les = self.config["VAE_LES"]["args"]
        print(config_les)
        vae_les_class =  self.config["VAE_LES"]["vae_class"]
        self.vae_les = MODEL_REGISTRY.get(vae_les_class)(**config_les)
        if self.config["VAE_LES"]["path_checkpoint"] is not None :
            checkpoint = torch.load(self.config["VAE_LES"]["path_checkpoint"], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
        x_saxs, recon_saxs, mu_saxs, logvar_saxs, z_saxs = self.vae_saxs(y=y_saxs, q=q_saxs, metadata=metadata)

        # Domaine LES
        y_les = batch["data_y_les"]
        q_les = batch["data_q_les"]
        x_les, recon_les, mu_les, logvar_les, z_les = self.vae_les(y=y_saxs, q=q_saxs, metadata=metadata)

        # Reconstructions croisées
        # SAXS→LES
        recon_saxs2les = self.vae_saxs.decode(z_les)
        # LES→SAXS
        recon_les2saxs = self.vae_les.decode(z_saxs)
        #recon_les2saxs = self.vae_saxs.decode(z_les)

        return {
            "recon_saxs": recon_saxs,
            "recon_les": recon_les,
            "recon_saxs2les": recon_saxs2les,
            "recon_les2saxs": recon_les2saxs,
            "z_saxs": z_saxs,
            "z_les": z_les
        }