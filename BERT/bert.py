import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, f_size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(f_size))
        self.beta = nn.Parameter(torch.zeros(f_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Residual(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class SingleHeadAttention(nn.Module):
    def forward(self, q, k, v, mask=False, dropout=None):
        ans = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(q.size(-1))
        if mask:
            ans = ans.masked_fill(mask == 0, -1e9)
        ans = F.softmax(ans, dim=-1)
        if dropout:
            ans = dropout(ans)
        return torch.matmul(ans, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.attn = SingleHeadAttention()
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=False):
        bs = q.size(0)
        q, k, v = [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear, (q, k, v))]
        ans = self.attn(q, k, v, mask, dropout=self.dropout)
        res = ans.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out_linear(res)

class PosEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model).float()
        pe.requires_grad = False
        pos = torch.arange(0, max_seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -torch.log(torch.tensor(10000.0)) / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.segment = SegmentEmbedding(embed_size)
        self.pos = PosEmbedding(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seg):
        ans = self.token(x) + self.segment(seg) + self.pos(x)
        return self.dropout(ans)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_FF, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_FF)
        self.linear2 = nn.Linear(d_FF, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = FeedForward(d_model=hidden, d_FF=feed_forward_hidden)
        self.residual1 = Residual(size=hidden, dropout=dropout)
        self.residual2 = Residual(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        return self.dropout(x)

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4
        self.dropout = dropout
        
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, seg):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, seg)
        for layer in self.layers:
            x = layer(x, mask)
        return x
