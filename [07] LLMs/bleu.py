from collections import Counter
import numpy as np
# removes punctuation and adds everything to lowercase and converts to list of list of str
def text_prep(sentences):
    if not type(sentences)== list:
        raise ValueError("Please enter a list")
    return [["".join(char.lower() for char in word if char.isalnum()) for word in sentence.split()]for sentence in sentences]

def ngram(candidate,references,n):
    if n<1:
        raise ValueError("Condition not met : N>=1")
    candidate_ngram = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
    max_ref=Counter()
    for ref in references:
        ref_ngram=Counter([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)])
        for n_gram in ref_ngram:
            max_ref[n_gram]=max(max_ref[n_gram],ref_ngram[n_gram])
    clipped_cnt={k:min(count,max_ref[k]) for k,count in candidate_ngram.items()}
    return sum(clipped_cnt.values()),sum(candidate_ngram.values())


def bp(candidate,references):
    if len(candidate)>min(len(ref) for ref in references):
        return 1
    else:
        return float(np.exp(1-len(candidate)/min(len(ref) for ref in references)))


def bleu(candidate_seq,reference_seq,n_max):
    candidate_seq: list[str] = text_prep([candidate_seq])[0]
    references_seq: list[list[str]] = text_prep(reference_seq)
    precision=[]
    for n in range(1,n_max+1):
        p_n,total=ngram(candidate_seq,reference_seq,n)
        precision.append(p_n/total if total>0 else 0)
    if all(p==0 for p in precision):
        return 0
    mean=np.exp(np.mean([np.log(p) for p in precision if p>0]))
    brev=bp(candidate_seq,reference_seq)
    return brev*mean
