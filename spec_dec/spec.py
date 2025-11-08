import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

class SpecDecoder:
    def __init__(self,small_model,big_model):
        print('loading small model')
        self.small_model=AutoModelForCausalLM.from_pretrained(small_model).eval()
        print('loading big model')
        self.big_model=AutoModelForCausalLM.from_pretrained(big_model).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(small_model)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.device='cuda'
        self.small_model.to(self.device)
        self.big_model.to(self.device)
    
    def generate_small(self,inp:torch.Tensor,n:int):
        inp=inp.to(self.device)
        gen_toks=[]
        gen_probs=[]
        for _ in range(n):
            with torch.no_grad():
                out=self.small_model(inp,use_cache=False)
                logits=out.logits[:,-1,:]
                probs=torch.softmax(logits,dim=-1)
                nxt=torch.multinomial(probs,num_samples=1)
                tok=nxt[:,0]
                gen_toks.append(tok.item())
                gen_probs.append(probs[0,tok].item())
                inp=torch.cat([inp,tok.unsqueeze(0)],dim=1)
        return gen_toks,gen_probs

    def verify(self,inp:torch.Tensor,gen_toks:List[int],gen_probs:List[float]):
        gen_seq=torch.cat([inp,torch.tensor([gen_toks],device=self.device)],dim=1)
        with torch.no_grad():
            out=self.big_model(gen_seq)
            logits=out.logits[0]
        final_tokens=[]
        seq_len=inp.size(1)
        for i in range(len(gen_toks)):
            pos=seq_len+i-1
            tar_probs=torch.softmax(logits[pos],dim=0)
            tar_prob=tar_probs[gen_toks[i]].item()
            gen_prob=gen_probs[i]
            acc_ratio=min(1.0,tar_prob/gen_prob)
            if torch.rand(1).item()<acc_ratio:
                final_tokens.append(gen_toks[i])
            else:
                adj_probs=torch.clamp(tar_probs-torch.softmax(logits[pos],dim=0),min=0.0)
                if adj_probs.sum()>0:
                    adj_probs=adj_probs/adj_probs.sum()
                    new_tok=torch.multinomial(adj_probs,num_samples=1).item()
                else:
                    new_tok=torch.multinomial(tar_probs,num_samples=1).item()
                final_tokens.append(new_tok)
                break
        if len(final_tokens)==len(gen_toks):
            pos=seq_len+i-1
            final_tokens.append(torch.multinomial(torch.softmax(logits[pos],dim=0),num_samples=1).item())
        return final_tokens
   
    def autoregressive_gen(self,prompt:str,max_new_tok:int):
        inp=self.tokenizer.encode(prompt,return_tensors='pt').to(self.device)
        with torch.no_grad():
            out=self.big_model.generate(inp,max_new_tokens=max_new_tok,do_sample=True,pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0],skip_special_tokens=True)
    
    def spec_gen(self,prompt:str,max_new_tok:int,num_sample_tok:int):
        inp=self.tokenizer.encode(prompt,return_tensors='pt').to(self.device)
        num_gen_tok=0
        it=0
        acc_total=0
        while num_gen_tok<max_new_tok:
            it+=1
            gen_toks,gen_probs=self.generate_small(inp,num_sample_tok)
            acc_toks=self.verify(inp,gen_toks,gen_probs)
            num_acc=len(acc_toks)
            gen_tokens+=num_acc
            acc_total+=num_acc
            inp=torch.cat([inp,torch.tensor([acc_toks],device=self.device)],dim=1)
            if num_gen_tok>=max_new_tok:
                break
        return self.tokenizer.decode(inp[0],skip_special_tokens=True)




