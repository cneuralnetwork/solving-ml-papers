import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def forward(self,q,k,v,mask=False,dropout=None):
        ans=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(q.size(-1))
        if mask:
            ans=ans.masked_fill(mask==0,-1e9)
        ans=F.softmax(ans,dim=-1)
        if dropout:
            ans=dropout(ans)
        return torch.matmul(ans,v)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model%h==0
        self.h=h
        self.d_k=d_model//h
        self.attn=SingleHeadAttention()
        self.linear=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(3)])
        self.out_linear=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,q,k,v,mask=False):
        bs=q.size(0)
        q,k,v=[l(x).view(bs,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linear,(q,k,v))]
        ans=self.attn(q,k,v,mask,dropout=self.dropout)
        res = ans.transpose(1,2).contiguous().view(bs,-1,self.h*self.d_k)
        return self.out_linear(res)
        