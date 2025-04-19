import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from model.residual import Residual
from model.feedforward import FeedForward
from model.embeddings import BERTEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = FeedForward(d_model=hidden, d_FF=feed_forward_hidden)
        self.residual1 = Residual(size=hidden, dropout=dropout)
        self.residual2 = Residual(size=hidden, dropout=dropout)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x, mask):
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        return self.dropout(x)
    
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=28, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads=attn_heads
        self.feed_forward_hidden=hidden*4
        self.dropout=dropout
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(hidden, attn_heads, hidden*4, dropout) for _ in range(n_layers)])
    def forward(self, x, seg):
        mask=(x>0).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)
        x=self.embedding(x,seg)
        for layer in self.layers:
            x=layer(x,mask)
        return x
    











