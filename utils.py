import torch
import matplotlib.pyplot as plt

import pytorch_lightning as pl

class InferencePlotCallback(pl.Callback):
    def __init__(self, dataloader, output_dir="inference_results"):
        super().__init__()
        self.dataloader = dataloader
        self.output_dir = output_dir

    def on_validation_end(self, trainer, pl_module):
        self.infer_and_plot(pl_module)

    def infer_and_plot(self, model,):
        """
        Perform inference and save reconstruction results in a single plot with subplots.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        model.eval().to("cuda")
        with torch.no_grad():
            batch = next(iter(self.dataloader))
            for el in batch:
                if isinstance(batch[el], torch.Tensor):
                    batch[el] = batch[el].to("cuda")
            data_y = batch["data_y"]
            inputs = data_y

            _, reconstructions, _, _ = model(batch)
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.ravel()

            for i in range(min(len(inputs), 4)):
                ax = axs[i]
                ax.plot(inputs[i].cpu().numpy(), label="Original")
                ax.plot(reconstructions[i].cpu().numpy(), label="Reconstructed")
                ax.set_title(f"Sample {i}")
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "combined_reconstructions.png")
            plt.savefig(plot_path)
            plt.close()
