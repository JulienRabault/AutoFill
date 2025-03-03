import torch
from torch import nn
import torch.nn.functional as F

class PairVAE(nn.Module):
    """
    Classe PairVAE pour l'entraînement par paire avec reconstruction croisée
    et alignement des espaces latents.
    """
    def __init__(self, vae_a, vae_b,
                 weight_latent=0.005,
                 weight_a_a=0.04,
                 weight_a_b=0.4,
                 weight_b_b=0.04,
                 weight_b_a=1.0):
        super(PairVAE, self).__init__()
        self.vae_a = vae_a
        self.vae_b = vae_b
        self.weight_latent = weight_latent
        self.weight_a_a = weight_a_a
        self.weight_a_b = weight_a_b
        self.weight_b_b = weight_b_b
        self.weight_b_a = weight_b_a

    def forward(self, batch):
        """
        Réalise l'encodage, la reconstruction et les reconstructions croisées.

        Paramètres:
            batch (dict): Contient :
                - "data_a" : images SAXS.
                - "data_b" : images SEM.

        Renvoie:
            dict: Dictionnaire contenant les reconstructions et variables latentes.
        """
        # Domaine A (SAXS)
        x_a = batch["data_a"]
        mu_a, logvar_a = self.vae_a.encode(x_a)
        z_a = self.vae_a.reparameterize(mu_a, logvar_a)
        recon_a = self.vae_a.decode(z_a)

        # Domaine B (SEM)
        x_b = batch["data_b"]
        mu_b, logvar_b = self.vae_b.encode(x_b)
        z_b = self.vae_b.reparameterize(mu_b, logvar_b)
        recon_b = self.vae_b.decode(z_b)

        # Reconstructions croisées
        # B→A
        cross_recon_a = self.vae_a.decode(z_b)
        # A→B
        cross_recon_b = self.vae_b.decode(z_a)

        return {
            "x_a": x_a,
            "x_b": x_b,
            "recon_a": recon_a,
            "recon_b": recon_b,
            "cross_recon_a": cross_recon_a,
            "cross_recon_b": cross_recon_b,
            "z_a": z_a,
            "z_b": z_b
        }

    @staticmethod
    def off_diagonal(x):
        """
        Retourne les éléments hors diagonale d'une matrice carrée.
        """
        n, m = x.shape
        assert n == m, "La matrice doit être carrée"
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_twins_loss(self, z_a, z_b):
        """
        Calcule la perte de similarité Barlow-Twins entre les espaces latents.

        """
        batch_size = z_a.size(0)
        z_a_flat = z_a.view(batch_size, -1)
        z_b_flat = z_b.view(batch_size, -1)

        z_a_norm = (z_a_flat - z_a_flat.mean(0)) / (z_a_flat.std(0) + 1e-9)
        z_b_norm = (z_b_flat - z_b_flat.mean(0)) / (z_b_flat.std(0) + 1e-9)

        c = torch.mm(z_a_norm.T, z_b_norm) / batch_size

        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()
        return on_diag + off_diag

    def compute_loss(self, batch):
        """
        Paramètres:
            batch (dict): Contient "data_a" (SAXS) et "data_b" (SEM).

        Renvoie:
            tuple: (loss_total, details) où details est un dictionnaire des pertes individuelles.
        """
        outputs = self.forward(batch)
        loss_a_a = F.mse_loss(outputs["recon_a"], batch["data_a"])
        loss_b_b = F.mse_loss(outputs["recon_b"], batch["data_b"])  # B→B
        loss_a_b = F.mse_loss(outputs["cross_recon_b"], batch["data_b"])  # A→B
        loss_b_a = F.mse_loss(outputs["cross_recon_a"], batch["data_a"])  # B→A
        loss_latent = self.barlow_twins_loss(outputs["z_a"], outputs["z_b"])

        loss_total = (self.weight_latent * loss_latent +
                      self.weight_a_a * loss_a_a +
                      self.weight_a_b * loss_a_b +
                      self.weight_b_b * loss_b_b +
                      self.weight_b_a * loss_b_a)

        details = {
            "loss_latent": loss_latent.item(),
            "loss_a_a": loss_a_a.item(),
            "loss_a_b": loss_a_b.item(),
            "loss_b_b": loss_b_b.item(),
            "loss_b_a": loss_b_a.item()
        }
        return loss_total, details
