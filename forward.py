import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
x0 = mnist[0][0] # [1, 28, 28]

T = 200 # timesteps
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion_sample(x0, t):
    """
    x0: (batch_size, 1, 28, 28)
    t: (batch_size,) - timestep for each sample
    """
    noise = torch.randn_like(x0)
    
    # Ensure alpha_bar is on the correct device
    device = x0.device
    alpha_bar_t = alpha_bar.to(device)[t].view(-1, 1, 1, 1)

    sqrt_ab = alpha_bar_t.sqrt()
    sqrt_one_minus_ab = (1 - alpha_bar_t).sqrt()

    return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise