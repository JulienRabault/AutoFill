import torch
import torch.nn as nn

#https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/barlow-twins.html#Barlow-Twins-Loss

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_saxs, zles):

        batch_size, feature_dim = z_saxs.shape

        # N x D, where N is the batch size and D is output dim of projection head
        z_saxs_norm = (z_saxs - torch.mean(z_saxs, dim=0)) / torch.std(z_saxs, dim=0)
        zles_norm = (zles - torch.mean(zles, dim=0)) / torch.std(zles, dim=0)

        cross_corr = torch.matmul(z_saxs_norm.T, zles_norm) / batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag

