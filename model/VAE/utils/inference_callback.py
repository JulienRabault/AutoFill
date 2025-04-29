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
    def __init__(self, dataloader, artifact_file="plot.png", output_dir="inference_results", num_samples=4, every_n_epochs=10, use_loglog=False):
        super().__init__()
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.num_samples = num_samples
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
            data_y = batch["data_y"]
            inputs = data_y.squeeze(1)
            outputs = model(batch)
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if "recon" in key:
                        recon = value
                        if isinstance(recon, torch.Tensor) and recon.ndim > inputs.ndim:
                            recon = recon.squeeze(1)
                        self._plot_and_log(trainer, inputs, recon, key)
            else:
                recon = outputs[1] if isinstance(outputs, tuple) else outputs
                if isinstance(recon, torch.Tensor) and recon.ndim > inputs.ndim:
                    recon = recon.squeeze(1)
                self._plot_and_log(trainer, inputs, recon, "recon")
        model.train()

    def _plot_and_log(self, trainer, inputs, reconstructions, key):
        indices = sample(range(len(inputs)), min(self.num_samples, len(inputs)))
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()
        for idx in range(len(indices)):
            i = indices[idx]
            ax = axs[idx]
            if self.use_loglog:
                ax.loglog(inputs[i].cpu().numpy(), label="Original")
                ax.loglog(reconstructions[i].cpu().numpy(), label="Reconstructed")
            else:
                ax.plot(inputs[i].cpu().numpy(), label="Original")
                ax.plot(reconstructions[i].cpu().numpy(), label="Reconstructed")
            ax.set_title(f"{key} Sample {i}")
            ax.legend()
            ax.grid(True)
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
            






