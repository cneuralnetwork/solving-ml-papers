import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Disc(nn.Module):
    def __init__(self,in_features) -> None:
        super().__init__()
        self.discriminator=nn.Sequential(nn.Linear(in_features,128),nn.LeakyReLU(0.1),nn.Linear(128,1),nn.Sigmoid())
    def forward(self,x):
        return self.discriminator(x)
    
class Gen(nn.modules):
    def __init__(self,z_dim,img_dim):
        super.__init__()
        self.generator=nn.Sequential(nn.Linear(z_dim,256),nn.LeakyReLU(0.1),nn.Linear(256,img_dim),nn.Tanh())
    def forward(self,x):
        return self.generator(x)
    
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
lr=3e-4
z_dim=64
img_dim=784
batch=32
epochs=50

disc=Disc(img_dim).to(device)
gen=Gen(z_dim,img_dim).to(device)
fixed_noise=torch.randn((batch,z_dim)).to(device)
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,),(0.5,)])
dataset=datasets.MNIST(root="dataset/",transform=transforms,download=True)
loader=DataLoader(dataset,batch_size=batch,shuffle=True)
optim_disc=optim.Adam(disc.parameters(),lr=lr)
optim_gen=optim.Adam(gen.parameters(),lr=lr)
loss_fn=nn.BCELoss()
writer_r=SummaryWriter(f"runs/GAN/real")
writer_f=SummaryWriter(f"runs/GAN/fake")
step=0

for epoch in range(epochs):
    for batch_idx,(real,_) in enumerate(loader):
        real=real.view(-1,784).to(device)
        batch=real.shape[0]
        noise=torch.randn(batch,z_dim).to(device)
        fake=gen(noise)
        disc_real=disc(real).view(-1)
        lossD_real=loss_fn(disc_real,torch.ones_like(disc_real))
        disc_fake=disc(fake).view(-1)
        lossD_fake=loss_fn(disc_fake,torch.zeros_like(disc_fake))
        lossD=(lossD_fake+lossD_real)/2
        disc.zero_grad()
        lossD.backward()
        optim_disc.step()

        out=disc(fake).view(-1)
        lossG=loss_fn(out,torch.ones_like(out))
        gen.zero_grad()
        lossG.backward()
        optim_gen.step()

        if batch_idx==0:
            print(f"Epoch : {epoch}/{epochs} | Loss Disc : {lossD:.4f} | Loss Gen : {lossG:.4f}")
            with torch.no_grad():
                fake=gen(fixed_noise).reshape(-1,1,28,28)
                data=real.reshape(-1,1,28,28)
                img_grid_fake=torch.utils.make_grid(fake,normalize=True)
                img_grid_real=torch.utils.make_grid(data,normalize=True)
                writer_f.add_image("Fake Images",img_grid_fake,global_step=step)
                writer_r.add_image("Real Images",img_grid_real,global_step=step)
                step+=1