import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, f_size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(f_size))
        self.eps = eps 
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm
