import torch
from torch import nn
import torch.nn.functional as F


class PairVAE(nn.Module):
    def __init__(self, vae_a, vae_b, latent_loss_weight=1.0):
        """
        vae_a : instance de CustomizableVAE entraînée sur le dataset A (ex. SAXS)
        vae_b : instance de CustomizableVAE entraînée sur le dataset B (ex. autre modalité)
        latent_loss_weight : coefficient pour la loss d'alignement des espaces latents
        """
        super(PairVAE, self).__init__()
        self.vae_a = vae_a
        self.vae_b = vae_b
        self.latent_loss_weight = latent_loss_weight

    def forward(self, data_a, data_b, metadata=None):
        # Reconstruction solo
        _, recon_a, mu_a, logvar_a = self.vae_a(data_a, None, metadata)
        _, recon_b, mu_b, logvar_b = self.vae_b(data_b, None, metadata)

        # Reconstruction croisée
        z_a = self.vae_a.reparameterize(mu_a, logvar_a)
        cross_recon_b = self.vae_b.decode(z_a).squeeze(1)

        z_b = self.vae_b.reparameterize(mu_b, logvar_b)
        cross_recon_a = self.vae_a.decode(z_b).squeeze(1)

        return {
            'data_a': data_a,
            'recon_a': recon_a,
            'data_b': data_b,
            'recon_b': recon_b,
            'cross_recon_a': cross_recon_a,
            'cross_recon_b': cross_recon_b,
            'mu_a': mu_a,
            'logvar_a': logvar_a,
            'mu_b': mu_b,
            'logvar_b': logvar_b,
        }

    def compute_loss(self, outputs, criterion=F.mse_loss):
        """
        Compute the total loss combining:
        - Reconstruction losses (solo & cross)
        - KL divergence for each VAE
        - Latent space alignment loss
        """
        recon_loss_a = criterion(outputs['recon_a'], outputs['data_a'])
        recon_loss_b = criterion(outputs['recon_b'], outputs['data_b'])

        cross_recon_loss_a = criterion(outputs['cross_recon_a'], outputs['data_a'])
        cross_recon_loss_b = criterion(outputs['cross_recon_b'], outputs['data_b'])

        kl_loss_a = -0.5 * torch.sum(1 + outputs['logvar_a'] - outputs['mu_a'].pow(2) - outputs['logvar_a'].exp())
        kl_loss_b = -0.5 * torch.sum(1 + outputs['logvar_b'] - outputs['mu_b'].pow(2) - outputs['logvar_b'].exp())

        latent_align_loss = F.mse_loss(outputs['mu_a'], outputs['mu_b'])

        total_loss = (recon_loss_a + recon_loss_b +
                      cross_recon_loss_a + cross_recon_loss_b +
                      kl_loss_a + kl_loss_b +
                      self.latent_loss_weight * latent_align_loss)

        return total_loss