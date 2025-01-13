import torch
from torch import nn

from VAE_base import BaseVAE


class VAE_1D(BaseVAE):
    def __init__(self, input_dim, latent_dim, learning_rate=1e-3):
        """
        Classe fille de BaseVAE qui utilise uniquement le tenseur Q.

        Args:
            input_dim (int): Dimension de l'entrée (pad_size).
            latent_dim (int): Dimension de l'espace latent.
            learning_rate (float): Taux d'apprentissage pour l'optimiseur.
        """
        super(VAE_1D, self).__init__(learning_rate)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # Mu et LogVar
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        """
        Encode le tenseur d'entrée x dans l'espace latent.

        Args:
            x (torch.Tensor): Tenseur d'entrée de dimension [nbatch_size, pad_size].

        Returns:
            torch.Tensor, torch.Tensor: Moyenne (mu) et logarithme de la variance (logvar).
        """
        encoded = self.encoder(x)
        mu, logvar = encoded.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Effectue le reparamétrage pour échantillonner à partir de l'espace latent.

        Args:
            mu (torch.Tensor): Moyenne de la distribution latente.
            logvar (torch.Tensor): Logarithme de la variance de la distribution latente.

        Returns:
            torch.Tensor: Échantillon dans l'espace latent.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Décode un tenseur latent z dans l'espace d'entrée.

        Args:
            z (torch.Tensor): Tenseur latent de dimension [nbatch_size, latent_dim].

        Returns:
            torch.Tensor: Reconstruction de l'entrée.
        """
        return self.decoder(z)

    def forward(self, Q, Y=None, metadata=None):
        """
        Passe avant du modèle (encode -> reparameterize -> decode).

        Args:
            Q (torch.Tensor): Tenseur d'entrée de dimension [nbatch_size, pad_size].

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Reconstruction, moyenne et logvar.
        """
        mu, logvar = self.encode(Q)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return Q, recon, mu, logvar

