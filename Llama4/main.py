import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelArgs:
    dim: int=4096
    n_layers: int = 32
    n_heads: int = 32 
    n_kv_heads: Optional[int]=None #
    vocab_size: int = -1 
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float]= None
    norm_eps:float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None
    num_local_experts: int = 16
    num_experts_per_tok: int = 2
    moe_layers: List[int] = None

# def precompute_theta_pos_freqs(head_dim:int, seq_len:int, device:str,theta:float=10000.0):
#     assert(head_dim%2)==0
#     #head/2
#     theta_num=torch.arange(0,head_dim,2).float()
#     #head/2
#     theta=1/(theta**(theta_num/head_dim)).to(device)
#     #seq_len
#     m=torch.arange(seq_len,device=device)
#     #all combinations - (seq_len) outer (head/2) -> (seq_len,head/2)
#     freqs=torch.outer(m,theta).float()
#     #convert to polar
#     freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
#     return freqs_complex

# def rope(x:torch.Tensor, freqs_complex:torch.Tensor, device: str):
#     #(b,seq_len,h,head)->(b,seq_len,h,head/2)
#     x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
#     #(seq_len,head/2)->(1,seq_len,1,head/2)
#     freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
#     #(b,seq_len,h,head/2)*(1,seq_len,1,head/2)->(b,seq_len,h,head/2)
#     x_rotated=x_complex*freqs_complex
#     #(b,seq_len,h,head/2)->(b,seq_len,h,head/2,2)
#     x_out=torch.view_as_real(x_rotated)
#     #(b,seq_len,h,head/2)->(b,seq_len,h,head)
#     x_out=x_out.reshape(*x.shape)

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.eps=eps
        self.weights=nn.Parameter(torch.ones(dim))

    def _norm(self,x:torch.Tensor):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x:torch.Tensor):
        return self.weights*self._norm(x.float())

class L2Norm(nn.Module):
    def __init__(self, dim: int = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    batch_size,seq_len,n_kv_heads,head_dim=x.shape
    if n_rep==1:
        return x
    else:
        return(
            x[:,:,:,None,:].expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim).reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
        )

class TextExperts(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_experts = config.num_local_experts
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.expert_dim = hidden_dim
        self.hidden_size = config.dim
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        next_states = torch.bmm((up * F.silu(gate)), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states

class MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.dim
        self.num_experts = config.num_local_experts
        self.experts = TextExperts(config)
        self.router = nn.Linear(config.dim, config.num_local_experts, bias=False)
        self.shared_expert = FeedForward(config)  # Shared expert after MoE

    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states_flat).transpose(0, 1)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = torch.topk(router_logits.transpose(0, 1), self.top_k, dim=1)
        router_scores = (
            torch.full_like(router_logits.transpose(0, 1), float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        
        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device).view(1, -1).expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
        routed_in = torch.gather(
            input=hidden_states_flat,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        
        out = self.shared_expert(hidden_states)
        out_flat = out.view(-1, hidden_dim)
        out_flat.scatter_add_(dim=0, index=router_indices, src=routed_out.view(-1, hidden_dim))
        return out_flat.view(batch, seq_len, -1), router_scores

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dm = args.dim // args.n_heads
        self.attention = SelfAttention(args, layer_idx)
        self.is_moe_layer = False
        if args.moe_layers is not None and layer_idx in args.moe_layers:
            self.is_moe_layer = True
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = FeedForward(args)
            
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x),
            start_pos,
            freqs_complex
        )
        
        residual = h
        h = self.ffn_norm(h)
        
        if self.is_moe_layer:
            h, _ = self.feed_forward(h)
        else:
            h = self.feed_forward(h)
            
        out = residual + h
        return out

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.layer_idx = layer_idx
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.use_qk_norm = args.dim <= 4096  # Small model uses QK norm
        if self.use_qk_norm:
            self.qk_norm = L2Norm()
            
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        args.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # Removed unnecessary parentheses
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
    
        if self.use_qk_norm:
            xq = self.qk_norm(xq)
            xk = self.qk_norm(xk)
        
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]
        
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        out = torch.matmul(scores, values)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class LLama4(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "Set a vocab size please."

        if args.moe_layers is None:
            if args.dim > 4096: 
                args.moe_layers = [i for i in range(args.n_layers) if i % 2 == 0]  # Even layers are MoE
            else:
                args.moe_layers = []  
                
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            self.layers.append(EncoderBlock(args, layer_idx=i))
            
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only process one token per pass"
        
        h = self.tok_embeddings(tokens)
        freqs_complex = None  
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
            
        h = self.norm(h)
        out = self.output(h)
        
        return out.float()
