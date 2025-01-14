import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int=4096
    n_layers: int = 32 
    n_heads: int = 32 #heads for Q
    n_kv_heads: Optional[int]=None # heads for K, V
    vocab_size: int = -1 # will set later
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float]= None
    norm_eps:float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.eps=eps
        self.weights=nn.Parameter(torch.ones(dim))
    def _norm(self,x:torch.Tensor):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x:torch.Tensor):
        return self.weight*self._norm(x.float()).type_as(x)
    
def precompute_theta_pos_freqs(head_dim:int, seq_len:int, device:str,theta:float=10000.0):
    assert(head_dim%2)==0
    #head/2
    theta_num=torch.arange(0,head_dim,2).float()
    #head/2
    theta=1/(theta**(theta_num/head_dim)).to(device)
    #seq_len
    m=torch.arange(seq_len,device=device)
    #all combinations - (seq_len) outer (head/2) -> (seq_len,head/2)
    freqs=torch.outer(m,theta).float()
    #convert to polar
    freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex

def rope(x:torch.Tensor, freqs_complex:torch.Tensor, device: str):
    #(b,seq_len,h,head)->(b,seq_len,h,head/2)
    x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    #(seq_len,head/2)->(1,seq_len,1,head/2)
    freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
    #(b,seq_len,h,head/2)*(1,seq_len,1,head/2)->(b,seq_len,h,head/2)
    x_rotated=x_complex*freqs_complex
    #(b,seq_len,h,head/2)->(b,seq_len,h,head/2,2)
    x_out=torch.view_as_real(x_rotated)
    #(b,seq_len,h,head/2)->(b,seq_len,h,head)
    x_out=x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device=device)

class EncoderBlock(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dm=args.dim//args.n_heads
        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)
        self.attention_norm=RMSNorm(args.dim,eps=args.norm_eps)
        self.ffn_norm=RMSNorm(args.dim,eps=args.norm_eps)

    def forward(self, x:torch.Tensor,start_pos: int,freqs_complex:torch.Tensor):
        h=x+self.attention.forward(self.attention_norm(x),start_pos,freqs_complex)
        out=h+self.feed_forward.forward(self.ffn_norm(h))
        return out
    
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    batch_size,seq_len,n_kv_heads,head_dim=x.shape
    if n_rep==1:
        return x
    else:
        return(
            x[:,:,:,None,:].expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim).reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
        )
    
class SelfAttention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q=args.n_heads
        self.n_rep=self.n_heads_q//self.n_kv_heads
        self.head_dim=args.dim//args.n_heads

        self.wq=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wo=nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)

        self.cache_k=torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))
        args.cache_v=torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))
    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        batch_size,seq_len,_=x.shape() #(B,1,dim)
        xq=self.wq(x)
        xk=self.wk(x)
        xv=self.wv(x)
        xq=xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        xk=xq.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        xv=xq.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        xq=rope(xq,freqs_complex)
        xk=rope(xk,freqs_complex)
        self.cache_k[:batch_size,start_pos:start_pos+seq_len]=xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len]=xv
        keys=self.cache_k[:batch_size,0:start_pos+seq_len]
        values=self.cache_v[:batch_size,0:start_pos+seq_len]
        keys=repeat_kv(keys,self.n_rep)
        values=repeat_kv(values,self.n_rep)
        xq=xq.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        scores=torch.matmul(xq,keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)
        out=torch.matmul(scores,values)
        out=out.transpose(1,2).contiguous().view(batch_size,seq_len,-1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        hidden_dim=4*args.dim
        hidden_dim=int(2*hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim=int(args.ffn_dim_multiplier*hidden_dim)
        hidden=args.multiple_of*((hidden+args.multiple_of-1)//args.multiple_of)
        self.w1=nn.Linear(args.dim,hidden_dim,bias=False)
        self.w2=nn.Linear(hidden_dim,args.dim,bias=False)
        self.w3=nn.Linear(args.dim,hidden_dim,bias=False)
    def forward(self,x:torch.Tensor):
        swish=F.silu(self.w1(x))
        x_V=self.w3(x)
        x=swish*x_V
        x=self.w2(x)
        return x
    
class LLama2(nn.Module):
    def __init__(self, args: ModelArgs)->None:
        super().__init__()
        assert args.vocab_size!=-1, "Set a vocab size please."
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embeddings=nn.Embedding(self.vocab_size,args.dim)
        self.layers=nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        self.norm=RMSNorm(args.dim,eps=args.norm_eps)
        self.output=nn.Linear(args.dim,self.vocab_size,bias=False)
        self.freqs_complex=precompute_theta_pos_freqs(self.args.dim//self.args.n_heads,self.args.max_seq_len*2,device=self.args.device)
    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len=tokens.shape
        assert seq_len==1,"only process one token per pass"
        h=self.tok_embeddings(tokens)
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]
        for layer in self.layers:
            h=layer(h,start_pos,freqs_complex)
        h=self.norm(h)
        out=self.output(h)
        return out.float()