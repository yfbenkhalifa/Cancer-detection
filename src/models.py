import torch
import pandas as pd
import numpy as np


class PCDModel_1(torch.nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.Layer_1 = torch.nn.Linear(input_shape, 32)
        self.Layer_2 = torch.nn.Linear(32, 64)
        self.Layer_3 = torch.nn.Linear(64, 128)
        self.OutLayer = torch.nn.Linear(128, output_shape)

    def forward(self, x):
        return self.OutLayer(
            self.Layer_3(
            self.Layer_2(
            self.Layer_1(x))))
    