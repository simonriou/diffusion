from torch.utils.data import DataLoader
from model import DenoiseNet, loss_fn
import torch
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
T = 200  # timesteps

model = DenoiseNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

for epoch in range(5):
    for x, _ in dataloader:
        x = x.to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device)
        loss = loss_fn(model, x, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), 'denoise_model.pth')