# %%

import torch
import numpy as np
from deepinv.models import DiffUNet
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the pretrained diffusion model
model = DiffUNet(
    in_channels=3, out_channels=3, pretrained="download", large_model=False
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# DDPM sampling parameters
num_timesteps = 1000  # Number of diffusion steps
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

os.makedirs("./img/ffhq_samples", exist_ok=True)


def sample_ddpm(model, image_size=256, batch_size=4, channels=3, idx=0):
    """Generate samples using DDPM sampling process."""
    # Start from pure noise
    x = torch.randn(batch_size, channels, image_size, image_size).to(device)

    # Progressively denoise the samples
    for t in tqdm(reversed(range(num_timesteps))):
        t_batch = torch.full((batch_size,), t, dtype=torch.long).to(device)

        # No gradient needed for sampling
        with torch.no_grad():
            # Predict noise
            noise_pred = model(x, t_batch, type_t="timestep")[:, :channels, ...]

            # Sample from posterior
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Equation for x_{t-1} given x_t and predicted noise
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

            x = coef1 * (x - coef2 * noise_pred) + torch.sqrt(betas[t]) * noise

            # Save the intermediate sample at this timestep
            if t % 10 == 0:  # Save every 100 steps to avoid too many images
                for b in range(batch_size):
                    img = x[b, ...].cpu().detach().numpy().transpose(1, 2, 0)
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"./img/ffhq_samples/sample_{idx+b}_{t}.png",
                        img,
                    )

    x = torch.clamp(x, 0, 1)

    return x


# Generate samples
num_samples = 8
batch_size = 1
samples_list = []

for i in range(num_samples):
    batch_samples = sample_ddpm(model, batch_size=1, idx=i)
    samples_list.append(batch_samples)
