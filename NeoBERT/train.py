import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from model.BERT import BERT
from model.MLM_NSP import MLM_NSP


class Optim():
    def __init__(self,optimizer,d_model,warmup_steps):
        self.optimizer=optimizer
        self._step=0
        self.warmup_steps=warmup_steps
        self.init_lr=d_model**(-0.5)
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        self.update_lr()
        self.optimizer.step()
        self._step+=1
    def update_lr(self):
        lr=self.init_lr*min(self.step**(-0.5),self._step*self.warmup_steps**(-1.5)*self.step)
        for param_group in self.optimizer.param_groups:
            param_group['lr']=lr
    def load_state_dict(self,state_dict):
        self.optimizer.load_state_dict(state_dict)
    def state_dict(self):
        return self.optimizer.state_dict()
    
class Trainer:
    def __init__(self, bert: BERT, vocab_size:int, train_data: torch.utils.data.DataLoader, test_data: torch.utils.data.DataLoader, lr: float=2e-4, beta=(0.9,0.999),weight_decay:float=0.01, warmup_steps=10000, cuda=True, device=None, log_freq=10):
        self.device=device
        if self.device is None:
            self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.bert=BERT
        self.model=MLM_NSP(self.bert,vocab_size).to(self.device)
        self.train_data=train_data
        self.test_data=test_data
        self.optim=Adam(self.model.parameters(),lr=lr,betas=beta,weight_decay=weight_decay)
        self.optim=Optim(self.optim,self.bert.hidden,warmup_steps)
        self.log_freq=log_freq
        self.lossfn=nn.NLLLoss(ignore_index=0)
    def train(self,epoch):
        self.iteration(epoch,self.train_data)
    def test(self,epoch):
        self.iteration(epoch,self.test_data,train=False)   
    def iteration(self,epoch,data_loader,train=True):
        str_code='train' if train else 'test'
        data_iter=tqdm.tqdm(enumerate(data_loader),f'{str_code} epoch {epoch}',total=len(data_loader),bar_format='{l_bar}{bar:10}{r_bar}')
        avg_loss=0.0
        corr, total=0,0
        for i, data in data_iter:
            data={key:value.to(self.device) for key,value in data.items()}
            next_sentence_output, mask_lm_output=self.model(data['bert_input'],data['segment_label'])
            next_loss=self.lossfn(next_sentence_output,data['is_next'])
            mask_loss=self.lossfn(mask_lm_output.transpose(1,2),data['bert_label'])
            loss=next_loss+mask_loss
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            cor=next_sentence_output.argmax(dim=-1).eq(data['is_nextl']).sum().item()
            avg_loss+=loss.item()
            corr+=cor
            total+=data['is_next'].nelement()
            data_iter.set_postfix({'epoch':epoch,'loss':loss.item(),'acc':cor/total})
        avg_loss/=len(data_iter)
        acc=corr/total
        print(f'{str_code} epoch {epoch} avg_loss={avg_loss} acc={acc}')
    def save(self,epoch,path):
        pass