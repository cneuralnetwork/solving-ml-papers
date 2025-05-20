from collections import Counter
import numpy as np

# removes punctuation and adds everything to lowercase and converts to list of list of str
def text_prep(sentences:list[str]):
    assert type(sentences)==list, "The input to the text-prep function should be a list of sentences."
    return [["".join(char.lower() for char in word if char.isalnum()) for word in sentence.split()]for sentence in sentences]

def ngram(candidate: list[str], references: list[list[str]], n:int):
    if n<1:
        raise ValueError("Condition not met : N>=1")
    candidate_ngram = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
    # print(candidate_ngram)
    max_ref=Counter()
    for ref in references:
        ref_ngram=Counter([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)])
        for n_gram in ref_ngram:
            max_ref[n_gram]=max(max_ref[n_gram],ref_ngram[n_gram])
    # print(max_ref)
    clipped_cnt={k:min(count,max_ref[k]) for k,count in candidate_ngram.items()}
    return sum(clipped_cnt.values()),sum(candidate_ngram.values())

def brevity_penalty(candidate:list[str],references:list[list[str]]):
    if len(candidate)>min(len(ref) for ref in references):
        return 1
    else:
        return float(np.exp(1-len(candidate)/min(len(ref) for ref in references)))

def bleu(candidate_seq: str, reference_seq: list[str], n_max: int):
    candidate_seq = text_prep([candidate_seq])[0]
    reference_seq = text_prep(reference_seq)
    precision=[]
    for n in range(1,n_max+1):
        p_n,total=ngram(candidate_seq,reference_seq,n)
        precision.append(p_n/total if total>0 else 0)
    # print(f"Precision = {precision}")
    if all(p==0 for p in precision):
        return 0
    mean=np.exp(np.mean([np.log(p) for p in precision if p>0]))
    bp=brevity_penalty(candidate_seq,reference_seq)
    return bp*mean

if __name__=="__main__":
    print(bleu("I have fun while playing cricket. I love cricket.",
               [
                   "I like to play cricket", 
                   "I enjoy playing cricket", 
                   "I play cicket and have fun", 
                   "I love to play cricket", 
                   "I love playing cricket"
                ],
                6))