import torch
from torch import nn
import torch.nn.functional as F

from VAE_base import BaseVAE



class CustomizableVAE(BaseVAE):
    def __init__(self, in_channels=1, out_channels=1, down_channels=[32, 64, 128], up_channels=[128, 64, 32], down_rate=[2, 2, 2], up_rate=[2, 2, 2], cross_attention_dim=16, learning_rate=1e-4, beta=0.001, strat="y"):
        super(CustomizableVAE, self).__init__(learning_rate, beta)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.cross_attention_dim = cross_attention_dim
        self.strat = strat

        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i in range(len(down_channels)):
            in_ch = in_channels if i == 0 else down_channels[i - 1]
            out_ch = down_channels[i]
            stride = down_rate[i]
            layer = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.ReLU(inplace=True)
            )
            self.encoder_layers.append(layer)

        # Bottleneck
        self.mu_layer = nn.Conv1d(down_channels[-1], cross_attention_dim, kernel_size=3, padding=1)
        self.logvar_layer = nn.Conv1d(down_channels[-1], cross_attention_dim, kernel_size=3, padding=1)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(up_channels)):
            in_ch = cross_attention_dim if i == 0 else up_channels[i - 1]
            out_ch = up_channels[i]
            layer = nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.decoder_layers.append(layer)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.ConvTranspose1d(up_channels[-1], out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """ Encodeur VAE """

        x = x.unsqueeze(1)
        for layer in self.encoder_layers:
            x = layer(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """ Reparamétrisation de l'espace latent """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """ Décodeur VAE """
        for i, layer in enumerate(self.decoder_layers):
            target_length = int(z.shape[2] * self.up_rate[i])
            z = F.interpolate(z, size=target_length, mode='nearest')
            z = layer(z)
        return self.output_layer(z)

    def forward(self, q, y, metadata):
        if self.strat == "q":
            x = q
        elif self.strat == "y":
            x = y
        else:
            raise ValueError("strat must be 'q' or 'y'")
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.squeeze(1)
        return x, recon, mu, logvar
