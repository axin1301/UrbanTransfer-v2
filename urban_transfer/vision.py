"""Trainable small visual encoder for initial tests; optional frozen feature backend later."""
import torch
from torch import nn
class SmallVisionEncoder(nn.Module):
    def __init__(self,out_dim=768):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,32,5,stride=2,padding=2),nn.BatchNorm2d(32),nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.GELU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,out_dim))
    def forward(self,x):return self.net(x)
