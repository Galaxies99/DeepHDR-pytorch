import torch
import numpy as np
import torch.nn as nn


class HDRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_img, ref_img):
        return torch.mean((out_img - ref_img) ** 2)
