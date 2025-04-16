import os
from random import sample

import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch

class InferencePlotCallback(pl.Callback):
    """
    Callback to perform inference and plot original and reconstructed outputs.
    If model returns a dict, plots are created for each key containing 'recon'.
    """
    def __init__(self, dataloader, artifact_file="plot.png", output_dir="inference_results", every_n_epochs=10, use_loglog=False):
        super().__init__()
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.use_loglog = use_loglog
        self.artifact_file = artifact_file
        

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.infer_and_plot(trainer, pl_module)

    def infer_and_plot(self, trainer, model):
        model.eval()
        device = next(model.parameters()).device
        with torch.inference_mode():
            batch = next(iter(self.dataloader))
            batch = move_to_device(batch, device)
            
            data_y_saxs = batch["data_y_saxs"].squeeze(1)
            data_y_les = batch["data_y_les"].squeeze(1)
            
            outputs = model(batch)
            self._plot_and_log(trainer, data_y_saxs, data_y_les, 
                               outputs["recon_saxs"].squeeze(1), outputs["recon_les"].squeeze(1), 
                               outputs["recon_saxs2les"].squeeze(1), outputs["recon_les2saxs"].squeeze(1), 
                               "recon")
            
        model.train()

    def _plot_and_log(self, trainer, data_y_saxs, data_y_les, 
                            recon_saxs, recon_les, recon_saxs2les, recon_les2saxs, key):
        
        indices = sample(range(len(data_y_saxs)), min(4, len(data_y_saxs)))
        fig, axs = plt.subplots(4, 2, figsize=(30, 10 *  min(4, len(data_y_saxs))))
        axs = axs.ravel()
        for idx in range(len(indices)):
            i = indices[idx]
            
            ax_saxs = axs[idx*2]
            ax_saxs.loglog(data_y_saxs[i].cpu().numpy(), label="SAXS")
            ax_saxs.loglog(recon_saxs[i].cpu().numpy(), label="Recon Saxs")
            ax_saxs.loglog(recon_les2saxs[i].cpu().numpy(), label="Les 2 Saxs")
            ax_saxs.set_title(f"{key} Sample SAXS {i}")
            ax_saxs.legend()
            ax_saxs.grid(True)
            
            ax_les = axs[idx*2+1]
            ax_les.plot(data_y_les[i].cpu().numpy(), label="LES")
            ax_les.plot(recon_les[i].cpu().numpy(), label="Recon Les")
            ax_les.plot(recon_saxs2les[i].cpu().numpy(), label="Saxs 2 Les") 
            ax_les.set_title(f"{key} Sample LES {i}")
            ax_les.legend()
            ax_les.grid(True)
            
        plt.tight_layout()

        if hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log_figure(trainer.logger.run_id, fig, artifact_file=self.artifact_file)

        else :
            os.makedirs(self.output_dir, exist_ok=True)
            key_dir = os.path.join(self.output_dir, "samples", key)
            os.makedirs(key_dir, exist_ok=True)
            plot_path = os.path.join(key_dir, f"epoch_{trainer.current_epoch}_{key}.png")
            plt.savefig(plot_path)
            plt.close()
            






