import torch

class RoPE():
    def __init__(self, head_dim, seq_len, device="cuda", theta=10000):
        super().__init__()
        self.head_dim=head_dim
        self.seq_len=seq_len
        self.device=device
        self.theta=theta
        self.freqs_complex=self.precompute_theta_pos_freq(head_dim,seq_len,device,theta)
    
    def forward(self,x):
        return self.rotary_pos_embeds(x,self.device)
    
    def precompute_theta_pos_freq(self,head_dim,seq,device,theta):
        assert head_dim%2==0 #dimension should be divisible by 2
        #theta_i=10000^(-2(i-1)/dim) for i={1,2,...,dim/2}
        theta_num=torch.arange(0,head_dim,2).float()
        theta=1./(theta**(theta_num/head_dim)).to(device)
        #write the m positions
        m=torch.arange(seq,device=device)
        #multiply m with every theta
        freqs=torch.outer(m,theta).float()
        #convert this into complex form (polar)
        freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
        return freqs_complex

    def rotary_pos_embeds(self, x,device):
        x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
        freqs_complex=self.freqs_complex.unsqueeze(0).unsqueeze(1)
        x_rotated=x_complex*freqs_complex
        x_out=torch.view_as_real(x_rotated)
        x_out=x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)
