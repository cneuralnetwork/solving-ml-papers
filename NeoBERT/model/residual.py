import torch
import torch.nn as nn
import torch.nn.functional as F
from model.RMS_Norm import RMSNorm

class Residual(nn.Module):
    def __init__(self,size,dropout):
        super().__init__()
        self.norm=RMSNorm(size)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))