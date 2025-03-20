import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # If input and output channels differ, use 1x1 convolution to match dimensions
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.deconv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # If input and output channels differ, match dimensions
        self.skip_connection = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.deconv1(x)
        out = self.relu(out)
        out = self.deconv2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out
        
class ResVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, in_channels=1,
                 down_channels=[32, 64, 128], up_channels=[128, 64, 32], dilation=1,
                 output_channels=1, strat="y", *args,  **kwargs):
        super(ResVAE, self).__init__()

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
            encoder_layers.append(ResidualBlock(current_in, out_ch, dilation))
            current_in = out_ch
            input_dim = input_dim // 2  # Since stride=2
        encoder_layers.append(nn.Flatten())
        flattened_size = down_channels[-1] * input_dim
        encoder_layers.append(nn.Linear(flattened_size, flattened_size // 2))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(flattened_size // 2, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size // 2, latent_dim)

        decoder_layers = []
        current_in = up_channels[0]
        for out_ch in up_channels[1:]:
            decoder_layers.append(ResidualUpBlock(current_in, out_ch))
            current_in = out_ch
        decoder_layers.append(nn.ConvTranspose1d(current_in, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, up_channels[0] * input_dim),
            nn.Unflatten(1, (up_channels[0], input_dim)),
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
        for i, layer in enumerate(self.decoder):
            z = layer(z)
        return z

    def forward(self, y, q=None, metadata=None):
        if self.strat == "y":
            x = y
        else:
            raise ValueError("strat must be 'q' or 'y'")

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return x, recon, mu, logvar, z