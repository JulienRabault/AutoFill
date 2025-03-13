import torch
from torch import nn
import torch.nn.functional as F

from VAE_base import BaseVAE



class CustomizableVAE(BaseVAE):
    def __init__(self, input_dim, latent_dim, in_channels=1,
                 down_channels=[32, 64, 128], up_channels=[128, 64, 32],
                 learning_rate=1e-3, output_channels=1, strat="y", beta=0.01):
        super(CustomizableVAE, self).__init__(learning_rate=learning_rate, beta=beta)
        if len(down_channels) != len(up_channels):
            raise ValueError("down_channels et up_channels doivent avoir la même taille.")
        if down_channels[-1] != up_channels[0]:
            raise ValueError("Le dernier canal de down_channels doit être égal au premier canal de up_channels.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.strat = strat

        encoder_layers = []
        current_in = in_channels
        for out_ch in down_channels:
            encoder_layers.append(nn.Conv1d(current_in, out_ch, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ReLU())
            current_in = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        test_tensor = torch.zeros(1, in_channels, input_dim)
        flattened_size = self.encoder(test_tensor).view(1, -1).size(1)
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)
        # self.fc_decode = nn.Linear(latent_dim, flattened_size)

        decoder_layers = []
        current_in = up_channels[0]

        self.decoder_input = nn.Linear(latent_dim, down_channels[-1] * input_dim)
        self.unflatten = nn.Unflatten(1, (down_channels[-1], input_dim))

        for i in range(len(up_channels)):
            if i < len(up_channels) - 1:
                out_ch = up_channels[i + 1]
                decoder_layers.append(nn.ConvTranspose1d(current_in, out_ch, kernel_size=3, stride=1, padding=1))
                decoder_layers.append(nn.ReLU())
                current_in = out_ch
            else:
                decoder_layers.append(
                    nn.ConvTranspose1d(current_in, output_channels, kernel_size=3, stride=1, padding=1))
                decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, down_channels[-1] * input_dim),
            self.unflatten,
            *decoder_layers
        )
        self.display_info()

    def display_info(self):
        test_tensor = torch.zeros(1, self.in_channels, self.input_dim)
        flattened_size = self.encoder(test_tensor).view(1, -1).size(1)
        print("VAE Architecture:")
        print("\tInput Dimension:", self.input_dim)
        print("\tLatent Dimension:", self.latent_dim)
        print("\tIn Channels:", self.in_channels)
        print("\tDown Channels:", self.down_channels)
        print("\tUp Channels:", self.up_channels)
        print("\tOutput Channels:", self.output_channels)
        print("\tFlattened Size:", flattened_size)
        print("\tEncoder Architecture:", self.encoder)
        print("\tDecoder Architecture:", self.decoder)


    def encode(self, x):
        """ Encodeur VAE """

        x = x.unsqueeze(1)
        for layer in self.encoder:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """ Reparamétrisation de l'espace latent """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """ Décodeur VAE """
        assert not torch.isnan(z).any(), "NaN detected in input latent variable z"

        for i, layer in enumerate(self.decoder):
            z = layer(z)
            assert not torch.isnan(z).any(), f"NaN detected after layer {i}"
        return z

    def forward(self, y, q=None, metadata=None):
        x=y
        assert not torch.isnan(x).any(), "NaN detected in input data"

        mu, logvar = self.encode(x)
        assert not torch.isnan(mu).any(), "NaN detected in mu"
        assert not torch.isnan(logvar).any(), "NaN detected in logvar"

        z = self.reparameterize(mu, logvar)
        assert not torch.isnan(z).any(), "NaN detected in latent variable z"

        recon = self.decode(z)
        assert not torch.isnan(recon).any(), "NaN detected in reconstructed output"

        recon = recon.squeeze(1)
        return x, recon, mu, logvar
