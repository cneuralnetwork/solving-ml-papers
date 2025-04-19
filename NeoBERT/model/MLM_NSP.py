import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BERT import BERT

class NSP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
    def forward(self, x):
        return F.log_softmax(self.linear(x[:,0]), dim=-1)
    
class MLM(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
    
class MLM_NSP(nn.Module):
    def __init__(self, BERT, vocab_size):
        super().__init__()
        self.bert = BERT()
        self.mlm = MLM(self.bert.hidden, vocab_size)
        self.nsp = NSP(self.bert.hidden)
    def forward(self, x, seg):
        x = self.bert(x, seg)
        return self.mlm(x), self.nsp(x)
