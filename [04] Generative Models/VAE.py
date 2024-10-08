# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_imag,make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

# Set dataset path
dataset_path='~/datasets'

# Set device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Set hyperparameters
batch=100
x_dim=784
hidden_dim=400
latent_dim=200
lr=10e-3
epochs=30

# Define MNIST dataset transformation
mnist_transform=transforms.Compose([transforms.ToTensor()])

# Set DataLoader parameters
kwargs={'num_workers':1,'pin_mem':True}

# Load MNIST dataset
train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch, shuffle=False, **kwargs)

# Define Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim,hidden_dim,latent_dim):
        super(Encoder,self).__init__()
        self.FC_in1=nn.Linear(input_dim,hidden_dim)
        self.FC_in2=nn.Linear(hidden_dim,hidden_dim)
        self.FC_mean=nn.Linear(hidden_dim,latent_dim)
        self.FC_var=nn.Linear(hidden_dim,latent_dim)
        self.LeakyReLU=nn.LeakyReLU(0.2)
        self.training=True
    
    def forward(self,x):
        fin=self.LeakyReLU(self.FC_in2(self.LeakyReLU(self.FC_in1(x))))
        mean=self.FC_mean(fin)
        logvar=self.FC_var(fin)
        return mean,logvar

# Define Decoder class
class Decoder(nn.Module):
    def __init__(self,latent_dim,hidden_dim,output_dim):
        super(Decoder,self).__init__()
        self.FC_h1=nn.Linear(latent_dim,hidden_dim)
        self.FC_h2=nn.Linear(hidden_dim,hidden_dim)
        self.FC_out=nn.Linear(hidden_dim,output_dim)
        self.LeakyReLU=nn.LeakyReLU(0.2)
    
    def forward(self,x):
        x_h=torch.sigmoid(self.FC_out(self.LeakyReLU(self.FC_h2(self.LeakyReLU(self.FC_h1(x))))))
        return x_h

# Define VAE class
class VAE(nn.Module):
    def __init__(self,Encoder,Decoder):
        super(VAE,self).__init__()
        self.Encoder=Encoder
        self.Decoder=Decoder
    
    def reparams(self,mean,var):
        e=torch.randn_like(var).to(device)
        z=mean+var*e
        return z
    
    def forward(self,x):
        mean,logvar=self.Encoder(x)
        z=self.reparams(mean,torch.exp(0.5*logvar))
        x_h=self.Decoder(z)
        return x_h,mean,logvar

# Initialize encoder, decoder, and VAE model
encoder=Encoder(input_dim=x_dim,hidden_dim=hidden_dim,latent_dim=latent_dim)
decoder=Decoder(latent_dim=latent_dim,hidden_dim=hidden_dim,output_dim=x_dim)
model=VAE(Encoder=encoder,Decoder=decoder).to(device)

# Define loss function
BCE_loss=nn.BCELoss()
def loss_fn(x,x_h,mean,logvar):
    rep_loss=nn.functional.binary_cross_entropy(x_h,x,reduction='sum')
    KLD=-0.5*torch.sum(1+logvar-mean.pow(2)-logvar.exp())
    return rep_loss+KLD

# Initialize optimizer
optim=Adam(model.parameters(),lr=lr)

# Training loop
model.train()
for ep in range(epochs):
    tot_loss=0
    for batch_idx,(x,_) in enumerate(train_loader):
        x=x.view(batch,x_dim)
        x=x.to(device)
        optim.zero_grad()
        x_h,mean,logvar=model(x)
        loss=loss_fn(x,x_h,mean,logvar)
        loss.backward()
        tot_loss+=loss.item()
        optim.step()
    print(f"Epoch {ep+1} | Loss : {tot_loss/(batch_idx*batch)}")
