import torch
import torch.nn as nn


class ChannelAdaptiveMixing(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        # x: bs * seq_len * c
