import torch
from torch import nn

from VAE_base import BaseVAE


# region VAE_conv1D

class VAE_conv1D(BaseVAE):
    def __init__(self, input_dim=2, latent_dim=64, hidden_dim=128, learning_rate=1e-3):
        super(VAE_conv1D, self).__init__(learning_rate)

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),

            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Réduction à un vecteur [batch, hidden_dim * 4, 1]
        )

        # Paramètres de la distribution latente
        self.fc_mu = nn.Linear(hidden_dim * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4, latent_dim)

        # Décodeur
        self.decoder_input = nn.Linear(latent_dim, hidden_dim * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(hidden_dim * 4 // 16, 16)),  # Ajuste à [batch, hidden_dim * 4 // 16, 16]

            nn.ConvTranspose1d(hidden_dim * 4 // 16, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Restriction dans [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        Applique le trick de reparamétrisation pour échantillonner.

        Args:
            mu (torch.Tensor): Moyenne.
            logvar (torch.Tensor): Log de la variance.

        Returns:
            torch.Tensor: Échantillon de la distribution latente.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,q, y, metadata):
        """
        Effectue un passage avant à travers le VAE.

        Args:
            Q (torch.Tensor): Entrée Q de forme [batch_size, 1, sequence_length].
            Y (torch.Tensor): Entrée Y de forme [batch_size, 1, sequence_length].
            metadata (torch.Tensor): Métadonnées associées (non utilisées ici).

        Returns:
            tuple: Reconstructions, moyenne (mu), et log-variance (logvar).
        """
        if Q.dim() == 2:
            Q = Q.unsqueeze(1)
        if Y.dim() == 2:
            Y = Y.unsqueeze(1)
        input_data = torch.cat((Q, Y), dim=1)
        mu, logvar = self.encode(input_data)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return input_data, recon, mu, logvar

    def encode(self, x):
        encoded = self.encoder(x).squeeze(-1)
        encoded_flattened = encoded.view(encoded.size(0), -1)
        mu = self.fc_mu(encoded_flattened)
        logvar = self.fc_logvar(encoded_flattened)
        return mu, logvar

    def decode(self, z):
        decoded = self.decoder_input(z)
        decoded = self.decoder(decoded)
        return decoded
# endregion