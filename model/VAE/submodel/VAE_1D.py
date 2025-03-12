import torch
from torch import nn


class VAE_1D(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE_1D, self).__init__()

        # Encoder avec une couche supplémentaire
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )

        # Decoder avec une couche supplémentaire
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),  # Nouvelle couche ajoutée
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self,q, y, metadata):
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return y, recon, mu, logvar

