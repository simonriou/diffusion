import torch.nn as nn
from forward import forward_diffusion_sample
import torch.nn.functional as F

class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x, t):
        t = t.float() / T
        t = t[:, None, None, None].expand_as(x)
        x = torch.cat([x, t], dim=1)
        return self.net(x)

def loss_fn(model, x0, t):
    x_t, noise = forward_diffusion_sample(x0, t)
    noise_pred = model(x_t, t)
    return F.mse_loss(noise_pred, noise)