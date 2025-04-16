import lightning.pytorch as pl
import torch.nn as nn
import torch

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

class MAEMetricCallback(pl.Callback):
    """Callback to compute and log MAE for each artifact with 'recon' in its key on validation set."""
    def __init__(self, val_dataloader):
        self.val_dataloader = val_dataloader
        self.mae_loss = nn.L1Loss()
        self.best_mae = {}  # Stores best MAE for each recon key

    def on_validation_epoch_end(self, trainer, pl_module):
        mae_dict = {}
        pl_module.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = move_to_device(batch, pl_module.device)
                data_y = batch["data_y"]
                inputs = data_y.squeeze(1)
                outputs = pl_module(batch)
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if "recon" in key:
                            recon = value
                            if isinstance(recon, torch.Tensor) and recon.ndim > inputs.ndim:
                                recon = recon.squeeze(1)
                            mae_value = self.mae_loss(recon, inputs).item()
                            mae_dict.setdefault(key, []).append(mae_value)
                else:
                    # Fallback if output is not dict, assume key "recon"
                    recon = outputs[1] if isinstance(outputs, tuple) else outputs
                    if isinstance(recon, torch.Tensor) and recon.ndim > inputs.ndim:
                        recon = recon.squeeze(1)
                    mae_value = self.mae_loss(recon, inputs).item()
                    mae_dict.setdefault("recon", []).append(mae_value)
        metrics = {}
        for key, values in mae_dict.items():
            avg_mae = sum(values) / len(values)
            if key not in self.best_mae or avg_mae < self.best_mae[key]:
                self.best_mae[key] = avg_mae
            metrics[f"val_mae_{key}"] = avg_mae
            metrics[f"best_val_mae_{key}"] = self.best_mae[key]
        trainer.logger.log_metrics(metrics, step=trainer.current_epoch)
        pl_module.train()
