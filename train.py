from torch.utils.data import DataLoader
from model import DenoiseNet, loss_fn
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

# Constants
T = 200  # total timesteps
EPOCHS = 20

# Model and optimizer
model = DenoiseNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0.0
    for batch_idx, (x, _) in enumerate(tqdm(dataloader, desc="Training")):
        x = x.to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device)

        loss = loss_fn(model, x, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {batch_loss:.4f}")

    print(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), 'denoise_model.pth')
print("\nModel saved to 'denoise_model.pth'")