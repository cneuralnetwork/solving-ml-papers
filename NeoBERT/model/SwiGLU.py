import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
    
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)
