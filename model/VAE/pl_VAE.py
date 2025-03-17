import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

import lightning.pytorch as pl

from model.VAE.submodel.registry import *


class PlVAE(pl.LightningModule):
    def __init__(self, config):
        super(PlVAE, self).__init__()

        self.config = config

        self.beta = config["beta"]

        model_class =  self.config["vae_class"]
        print(self.config)
        self.model = MODEL_REGISTRY.get(model_class)(**self.config)

    def compute_loss(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kl_div, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        y = batch["data_y"]
        q = batch["data_q"]
        metadata = batch["metadata"]
        
        x, recon, mu, logvar, z = self.model(y=y, q=q, metadata=metadata)
        loss, recon_loss, kl_loss = self.compute_loss(x, recon, mu, logvar)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["data_y"]
        q = batch["data_q"]
        metadata = batch["metadata"]
        
        x, recon, mu, logvar, z = self.model(y=y, q=q, metadata=metadata)
        loss, recon_loss, kl_loss = self.compute_loss(x, recon, mu, logvar)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["max_lr"])
              
        scheduler = CosineAnnealingLR(
          optimizer,
          T_max=self.config["T_max"],
          eta_min=self.config["eta_min"])

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"}}

