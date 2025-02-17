import torch
from torch import nn

from VAE_base import BaseVAE


class Conv1DVAE_concat(BaseVAE):
    def __init__(self, input_dim, latent_dim, learning_rate=1e-3, **kwargs):
        super(Conv1DVAE_concat, self).__init__(learning_rate=learning_rate, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),  # (bs, 32, input_dim)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # (bs, 64, input_dim)
            nn.ReLU(),
            nn.Flatten()
        )
        # Calculer dynamiquement la taille après les convolutions
        test_tensor = torch.zeros(1, 1, input_dim)  # Tensor fictif pour calculer la sortie
        flattened_size = self.encoder(test_tensor).view(1, -1).size(1)

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # (bs, 32, input_dim)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),  # (bs, 1, input_dim)
            nn.Sigmoid()
        )


    def encode(self, x):
        x = x.unsqueeze(1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 64, -1)  # (bs, 64, input_dim/4)
        recon = self.decoder(h)  # (bs, 1, input_dim)
        return recon.squeeze(1)  # (bs, input_dim)

    def forward(self, q, y, metadata):
        # x = q + y
        x = y
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return x, recon, mu, logvar
