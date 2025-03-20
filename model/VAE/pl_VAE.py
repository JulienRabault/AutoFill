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

        self.beta = config["training"]["beta"]

        model_class =  self.config["model"]["vae_class"]
        self.output_transform_log = self.config["model"]["output_transform_log"]
        
        self.model = MODEL_REGISTRY.get(model_class)(**self.config["model"]["args"])

    def forward(self, x):
        y = x["data_y"]
        q = x["data_q"]
        metadata = x["metadata"]
        return self.model(y=y, q=q, metadata=metadata)
        
    def compute_loss(self, x, recon, mu, logvar):
        if self.output_transform_log :
            x = torch.log(x + 1e-9)
            recon = torch.log(recon+ 1e-9)
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
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["data_y"]
        q = batch["data_q"]
        metadata = batch["metadata"]
        
        x, recon, mu, logvar, z = self.model(y=y, q=q, metadata=metadata)
        loss, recon_loss, kl_loss = self.compute_loss(x, recon, mu, logvar)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        
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

