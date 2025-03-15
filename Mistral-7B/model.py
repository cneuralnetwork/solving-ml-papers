import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,Tuple

class ModelArgs:
    def __init__(self, dim:int, n_layers:int, head_dim:int, hidden_dim:int, n_heads:int, n_kv_head:int, norm_eps:int, vocab_size:int, rope_theta:float=10000):
        self.dim=dim
        self.n_layers=n_layers
        self.head_dim=head_dim
        self.hidden_dim=hidden_dim
        self.n_heads=n_heads
        self.n_kv_head=n_kv_head
        self.norm_eps=norm_eps
        self.vocab_size=vocab_size
        self.rope_theta=rope_theta

class FFN(nn.Module):
    def __init__(self, args="ModelArgs"):
        super().__init__()
        self.l1=nn.Linear(args.dim, args.hidden_dim,bias=False)
        self.l2=nn.Linear(args.hidden_dim, args.dim,bias=False)
        self.l3=nn.Linear(args.dim, args.hidden_dim,bias=False)
    def forward(self,x)-> torch.Tensor:
        # out=l2(silu(l1(x).l3))
        return self.l2(F.silu(self.l1(x))+self.l3(x))
    
class RMSNorm(nn.Module):
    def __init__(self, dims:int, eps:float=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(dims))
        self.eps=eps
    def mid(self,x):
        # rec sqrt of rms 
        k=x*((x**2).mean(-1,keepdim=True)+self.eps).rsqrt()
        return k
    def forward(self,x):
        # final output
        return self.weight*self.mid(x.float()).type(x.dtype)

class rope(nn.Module):
    def __init__(self, dim:int, b:float=10000):
        super().__init__()
        self.dim=dim
        self.b=b
        self.freqs=self.fr(dim//2)
    def fr(self, n:int):
        # freq=1/base^(2i/dim)
        return 1.0/(10000**(torch.arange(0,n,2)/n))
    def forward(self,x:torch.Tensor, offset:int=0):
        # position indices
        t=torch.arange(x.shape[2],device="cuda")+offset
        freqs=self.freqs.to("cuda")
        # [tθ_0, tθ_1, ..., tθ_{d/2-1}]
        t_sin=torch.sin(t[:,None]*freqs[None,:])
        t_cos=torch.cos(t[:,None]*freqs[None,:])
        # x' = x.cosθ + x.sinθ
        ans=torch.stack([x[...,0::2]*t_cos+x[...,1::2]*t_sin #even
                         , -x[...,0::2]*t_sin+x[...,1::2]*t_cos] #odd
                         ,dim=-1).flatten(-2,-1)
        return ans        

class attn(nn.Module):
    def __init__(self, args="ModelArgs"):
        super().__init__()
        self.args=args
        self.n_heads=args.n_heads
        self.n_kv_head=args.n_kv_head
        self.repeats=self.n_heads//self.n_kv_head #GQA
        self.scale=self.args.head_dim**-0.5
        self.wq=nn.Linear(args.dim, args.n_heads*args.head_dim, bias=False)
        self.wk=nn.Linear(args.dim, args.n_kv_heads*args.head_dim, bias=False)
        self.wv=nn.Linear(args.dim, args.n_kv_heads*args.head_dim, bias=False)
        self.wo=nn.Linear(args.n_heads*args.head_dim, args.dim, bias=False)
        self.rope=rope(args.head_dim,args.rope_theta)
    def forward(self,x:torch.Tensor, mask: Optional[torch.Tensor]=None,
                cache:Optional[Tuple[torch.Tensor, torch.Tensor]]=None)->torch.Tensor:
        # q,k,v
        q=self.wq(x)
        k=self.wk(x)
        v=self.wv(x)
        # reshape to [b,nh,l,hd]
        b,l,d=x.shape
        q=q.view(b,l,self.n_heads,-1).transpose(1,2)
        k=k.view(b,l,self.n_kv_head,-1).transpose(1,2)
        v=v.view(b,l,self.n_kv_head,-1).transpose(1,2)
        # repeat for gqa
        def repeat(c):
            return torch.cat([c.unsqueeze(2)]*self.repeats,dim=2).view([b,self.n_heads,l,-1])
        k,v=map(repeat,[k,v])
        # cache
        if cache is not None:
            key_c,val_c=cache
            q=self.rope(q,offset=key_c.shape[2])
            k=self.rope(k,offset=key_c.shape[2])
            k=torch.cat([key_c,k],dim=2)
            v=torch.cat([val_c,v],dim=2)
        else:
            q=self.rope(q)
            k=self.rope(k)
        # QK^T/√d_k
        atn=(q*self.scale)@k.transpose(-1,-2)
        # applymask (causal)
        if mask is not None:
            atn+=mask
        # softmax
        atn=F.softmax(atn.float(),dim=-1).type(atn.dtype)
        out=(atn@v).transpose(1,2).contiguous().reshape(b,l,-1)
        return self.wo(out),(k,v)

class transformer(nn.Module):
    def __init__(self, args="ModelArgs"):
        super().__init__()
        self.attn=attn(args)
        self.ff=FFN(args)
        self.attn_norm=RMSNorm(args.dim,args.norm_eps)
        self.ff_norm=RMSNorm(args.dim,args.norm_eps)
    def forward(self,x:torch.Tensor, mask:Optional[torch.Tensor]=None,
                cache:Optional[Tuple[torch.Tensor, torch.Tensor]]=None)->torch.Tensor:
        #residual
        r,cache=self.attn(self.attn_norm(x),mask,cache)
        x=x+r
        r=self.ff(self.ff_norm(x))
        ans=x+r
        return ans,cache

class mistral(nn.Module):
    def __init__(self, args="ModelArgs"):
        super().__init__()
        assert args.vocab_size>0    
        self.tok_embedding=nn.Embedding(args.vocab_size,args.dim)
        self.layer=nn.ModuleList([transformer(args) for _ in range(args.n_layers)])
        self.norm=RMSNorm(args.dim,args.norm_eps)
        self.out=nn.Linear(args.dim,args.vocab_size,bias=False) #Unembedding
        def gen_mask(self,s):
            return torch.triu(torch.ones(s,s)*float("-inf"),diagonal=1)
        def forward(self,input:torch.Tensor,cache=None):
            h=self.token_embedding(input)
            m=None
            if h.shape[1]>1:
                m=self.gen_mask(h.shape[1]).to("cuda")
            if cache is None:
                cache=[None]*len(self.layer)
            for e,layer in enumerate(self.layer):
                h,cache[e]=layer(h,m,cache[e])
            return self.out(self.norm(h)),cache
            


        