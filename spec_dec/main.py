from spec import *

decode=SpecDecoder('gpt2','gpt2-xl')
print(decode.autoregressive_gen("What is 2+2?",max_new_tok=50))
