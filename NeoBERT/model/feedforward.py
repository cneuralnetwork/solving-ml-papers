import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SwiGLU import SwiGLU

class FeedForward(nn.Module):
    def __init__(self, d_model, d_FF,dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_FF)
        self.linear2 = nn.Linear(d_FF, d_model)
        self.dropout = nn.Dropout(dropout)
        self.swiglu=SwiGLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.swiglu(self.linear1(x))))