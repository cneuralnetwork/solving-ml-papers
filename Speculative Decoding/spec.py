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
        gen_distributions=[]
        for _ in range(n):
            with torch.no_grad():
                out=self.small_model(inp,use_cache=True)
                logits=out.logits[:,-1,:]
                probs=torch.softmax(logits,dim=-1)
                nxt=torch.multinomial(probs,num_samples=1)
                tok=nxt[:,0]
                gen_toks.append(tok.item())
                gen_distributions.append(probs)
                inp=torch.cat([inp,tok.unsqueeze(0)],dim=1)
        return gen_toks,gen_distributions

    def verify(self,inp:torch.Tensor,gen_toks:List[int],small_model_distributions:List[torch.Tensor]):
        gen_toks_tensor=torch.tensor([gen_toks],device=self.device)
        gen_seq=torch.cat([inp,gen_toks_tensor],dim=1)
        with torch.no_grad():
            out=self.big_model(gen_seq)
            logits=out.logits[0]
        seq_len=inp.size(1)
        tgt_logits=logits[seq_len-1:-1,:]
        tgt_probs=torch.softmax(tgt_logits,dim=-1)
        small_dist=torch.cat(small_model_distributions,dim=0)
        tok_idx=gen_toks_tensor.T
        tgt_probs_gen=torch.gather(tgt_probs,1,tok_idx).squeeze(-1)
        small_probs_gen=torch.gather(small_dist,1,tok_idx).squeeze(-1)
        ratio=tgt_probs_gen/small_probs_gen
        rand_vals=torch.rand(len(gen_toks),device=self.device)
        accept_mask=rand_vals<ratio
        rej_ids=(~accept_mask).nonzero()
        if rej_ids.numel()>0:
            rej_idx=rej_ids[0].item()
            final=gen_toks[:rej_idx]
            pos=seq_len+rej_idx-1
            big_probs=torch.softmax(logits[pos],dim=0)
            small_probs=small_dist[rej_idx]
            diff_probs=torch.clamp(big_probs-small_probs,min=0.0)
            if diff_probs.sum()>0:
                diff_probs=diff_probs/diff_probs.sum()
                new_tok=torch.multinomial(diff_probs,num_samples=1).item()
            else:
                new_tok=torch.multinomial(big_probs,num_samples=1).item()
            final.append(new_tok)
        else:
            final=gen_toks
            pos=seq_len+len(gen_toks)-1
            last_probs=torch.softmax(logits[pos],dim=0)
            final.append(torch.multinomial(last_probs,num_samples=1).item())
        return final
   
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
            gen_toks,gen_distributions=self.generate_small(inp,num_sample_tok)
            acc_toks=self.verify(inp,gen_toks,gen_distributions)
            num_acc=len(acc_toks)
            num_gen_tok+=num_acc
            acc_total+=num_acc
            inp=torch.cat([inp,torch.tensor([acc_toks],device=self.device)],dim=1)
            if num_gen_tok>=max_new_tok:
                break
        return self.tokenizer.decode(inp[0],skip_special_tokens=True)




