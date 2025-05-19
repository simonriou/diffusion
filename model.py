import torch.nn as nn
import torch
from forward import forward_diffusion_sample
import torch.nn.functional as F

T = 200

class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x, t):
        # t: [batch] -> [batch, 1]
        t = t.float().unsqueeze(-1) / T
        t_embed = self.time_embed(t)  # [batch, 128]

        # Inject t into conv layers (e.g. using FiLM or channel-wise add)
        # Simplest: just ignore it for now or return to 2-channel version

        return self.net(x)

def loss_fn(model, x0, t):
    x_t, noise = forward_diffusion_sample(x0, t)
    noise_pred = model(x_t, t)
    return F.mse_loss(noise_pred, noise)