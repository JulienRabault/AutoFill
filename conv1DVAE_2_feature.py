import torch
from torch import nn

from VAE_base import BaseVAE


class Conv1DVAE_2_feature(BaseVAE):
    def __init__(self, input_dim, latent_dim, learning_rate=1e-3):
        super(Conv1DVAE_2_feature, self).__init__(learning_rate=learning_rate)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encodeur avec 2 canaux (q et y traités ensemble)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculer la taille après encodage
        test_tensor = torch.zeros(1, 2, input_dim)
        flattened_size = self.encoder(test_tensor).view(1, -1).size(1)

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, q, y):
        x = torch.stack((q, y), dim=1)  # (bs, 2, input_dim)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 64, -1)
        recon = self.decoder(h)
        return recon.squeeze(1)

    def forward(self, q, y, metadata):
        mu, logvar = self.encode(q, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return q + y, recon, mu, logvar
