import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import DenoiseNet

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
x0 = mnist[0][0] # [1, 28, 28]

T = 200 # timesteps
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

@torch.no_grad()
def sample(model):
    img = torch.randn((1, 1, 28, 28), device=device)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=device)

        beta_t = beta[t].to(device)
        alpha_t = alpha[t].to(device)
        alpha_bar_t = alpha_bar[t].to(device)

        noise_pred = model(img, t_tensor)

        coef1 = 1 / alpha_t.sqrt()
        coef2 = (1 - alpha_t).sqrt() / alpha_bar_t.sqrt()

        img = coef1 * (img - coef2 * noise_pred)

        if t > 0:
            noise = torch.randn_like(img)
            img += beta_t.sqrt() * noise

    return img

model = DenoiseNet().to(device)
model.load_state_dict(torch.load("denoise_model.pth", map_location=device))
model.eval()

generated = sample(model)

plt.imshow(generated.squeeze().cpu(), cmap='gray')
plt.axis('off')
plt.title('Generated Digit')
plt.show()