import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pos = torch.arange(0, max_seq_len).float()
        pos_freq = torch.outer(pos, inv_freq) 
        cos_pos = torch.cos(pos_freq) 
        sin_pos = torch.sin(pos_freq) 
        self.register_buffer('cos_pos', cos_pos)
        self.register_buffer('sin_pos', sin_pos)
    
    def forward(self, x):
        seq_len = x.size(1)
        cos = self.cos_pos[:seq_len]  
        sin = self.sin_pos[:seq_len]  
        cos = cos.unsqueeze(0) 
        sin = sin.unsqueeze(0)  
        x_even = x[..., 0::2]  
        x_odd = x[..., 1::2]   
        rotated_x_even = x_even * cos - x_odd * sin
        rotated_x_odd = x_odd * cos + x_even * sin
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = rotated_x_even
        x_rotated[..., 1::2] = rotated_x_odd
        return x_rotated

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
        self.rope = RoPE(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, seg):
        combined = self.token(x) + self.segment(seg)
        position_encoded = self.rope(combined)
        return self.dropout(position_encoded)
