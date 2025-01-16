import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BaseVAE(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(BaseVAE, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

    def encode(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

    def decode(self, z):
        raise NotImplementedError("Subclasses must implement this method.")

    def reparameterize(self, mu, logvar):
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self,q, y, metadata):
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_loss(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
       q, y, metadata = separate_batch_elements(batch)
        x, recon, mu, logvar = self.forward(Q, Y, metadata)
        loss, recon_loss, kl_loss = self.compute_loss(x, recon, mu, logvar)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
       q, y, metadata = separate_batch_elements(batch)
        x, recon, mu, logvar = self.forward(Q, Y, metadata)
        loss, _, _ = self.compute_loss(x, recon, mu, logvar)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def separate_batch_elements(batch):
    """
    Sépare les éléments d'un batch en trois listes distinctes :q, y, et metadata.

    Args:
        batch (list): Liste de tuples contenant (Q, Y, metadata).

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: Tenseursq, y et metadata séparés.
    """
    Q_list, Y_list, metadata_list = zip(*batch)  # Décompresse les tuples en trois listes

    # Convertir les listes en tenseurs
    Q_tensor = torch.stack(Q_list, dim=0)
    Y_tensor = torch.stack(Y_list, dim=0)
    metadata_tensor = torch.stack(metadata_list, dim=0)

    return Q_tensor, Y_tensor, metadata_tensor
