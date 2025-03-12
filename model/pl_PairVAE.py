import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CyclicLR

import pytorch_lightning as pl

from model.PairVAE import PairVAE
from model.loss import BarlowTwinsLoss

class PlPairVAE(pl.LightningModule):
    """
    Classe PairVAE pour l'entraînement par paire avec reconstruction croisée
    et alignement des espaces latents.
    """
    def __init__(self, config):
        super(PlPairVAE, self).__init__()

        self.config = config
        
        self.pair_vae = PairVAE(self.config["model"])

        self.barlow_twins_loss = BarlowTwinsLoss(self.config["training"]["lambda_param"])

        self.weight_latent_similarity = self.config["training"]["weight_latent_similarity"]
        self.weight_kld_saxs = self.config["training"]["weight_kld_saxs"]
        self.weight_kld_les= self.config["training"]["weight_kld_les"]
        
        self.weight_saxs2saxs = self.config["training"]["weight_saxs2saxs"]
        self.weight_saxs2les = self.config["training"]["weight_saxs2les"]
        self.weight_les2les = self.config["training"]["weight_les2les"]
        self.weight_les2saxs = self.config["training"]["weight_les2saxs"]


    def compute_loss(self, batch, outputs):
        """
        Paramètres:
            batch (dict): Contient "data_saxs" (SAXS) et "data_b" (SEM).

        Renvoie:
            tuple: (loss_total, details) où details est un dictionnaire des pertes individuelles.
        """
        
        loss_saxs2saxs = F.mse_loss(outputs["recon_saxs"], batch["data_saxs"])
        loss_les2les = F.mse_loss(outputs["recon_les"], batch["data_les"]) 
        loss_saxs2les = F.mse_loss(outputs["recon_saxs2les"], batch["data_les"])  
        loss_les2sax = F.mse_loss(outputs["recon_les2saxs"], batch["data_saxs"])  

        loss_latent = self.barlow_twins_loss(outputs["z_saxs"], outputs["z_les"])

        # Kld loss ?

        loss_total = (self.weight_latent_similarity * loss_latent +
                      self.weight_saxs2saxs * loss_saxs2saxs +
                      self.weight_les2les* loss_les2les +
                      self.weight_saxs2les * loss_saxs2les +
                      self.weight_les2saxs * loss_les2sax)

        details = {
            "loss_latent": loss_latent.item(),
            "loss_saxs2saxs": loss_saxs2saxs.item(),
            "loss_les2les": loss_les2les.item(),
            "loss_saxs2les": loss_saxs2les.item(),
            "loss_les2sax": loss_les2sax.item()
        }
        return loss_total, details

    def training_step(self, batch, batch_idx):

        outputs = self.pair_vae.forward(batch)

        loss_total, details = self.compute_loss(batch, outputs)

        self.log('train_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_saxs2saxs', details["loss_saxs2saxs"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_loss_les2les', details["loss_les2les"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_loss_saxs2les', details["loss_saxs2les"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_loss_les2sax', details["loss_les2sax"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss_total

    def validation_step(self, batch, batch_idx):

        outputs = self.pair_vae.forward(batch)

        loss_total, details = self.compute_loss(batch, outputs)

        self.log('val_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)

        self.log('val_loss_saxs2saxs', details["loss_saxs2saxs"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_loss_les2les', details["loss_les2les"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_loss_saxs2les', details["loss_saxs2les"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_loss_les2sax', details["loss_les2sax"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.pair_vae.parameters(), lr=self.config["training"]["max_lr"])
              
        scheduler = CyclicLR(
          optimizer,
          base_lr=self.config["training"]["base_lr"],
          max_lr=self.config["training"]["max_lr"],     
          step_size_up=self.config["training"]["step_size_up"],   
          step_size_down=self.config["training"]["step_size_down"],
          mode="triangular",    
          cycle_momentum=False)

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"}}
