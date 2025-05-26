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
checkpoint_path = "./weights/model_final.pth"
model = DiffUNet(in_channels=1, out_channels=1, pretrained=Path(checkpoint_path))

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    print(f"Loaded model from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

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

os.makedirs("./img/mnist_samples", exist_ok=True)


def sample_ddpm(model, image_size=32, batch_size=4, channels=1):
    """Generate samples using DDPM sampling process."""
    # Start from pure noise
    x = torch.randn(batch_size, channels, image_size, image_size).to(device)

    # Progressively denoise the samples
    for t in tqdm(reversed(range(num_timesteps))):
        t_batch = torch.full((batch_size,), t, dtype=torch.long).to(device)

        # No gradient needed for sampling
        with torch.no_grad():
            # Predict noise
            noise_pred = model(x, t_batch, type_t="timestep")

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
                idx = 0  # Use this as a base index for the current batch
                for b in range(batch_size):
                    img = x[b, 0].cpu().detach().numpy()
                    # Min-max normalization to [0, 1] range
                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )  # Add small epsilon to avoid division by zero
                    plt.imsave(
                        f"./img/mnist_samples/sample_{idx+b}_{t}.png", img, cmap="gray"
                    )

    x = torch.clamp(x, 0, 1)

    return x


# Generate samples
num_samples = 8
batch_size = 8
samples_list = []

for i in range(num_samples // batch_size):
    batch_samples = sample_ddpm(model, batch_size=batch_size)
    samples_list.append(batch_samples)
