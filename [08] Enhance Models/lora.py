import torch
import torch.nn as nn

class LoRA(nn.Module):
  def __init__(self,in_f,out_f,rank=1,alpha=1,device:str="cpu"):
    super().__init__()
    self.A=nn.Parameter(torch.zeros((rank,out_f)).to(device))
    self.B=nn.Parameter(torch.zeros((in_f,rank)).to(device))
    self.scale=alpha/rank
    self.en=True
  def forward(self,wts):
    if self.en:
      return wts+torch.matmul(self.B,self.A).view(wts.shape)*self.scale
    else:
      return wts