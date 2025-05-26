# %%

import torch
from torch.utils.data import DataLoader
import numpy as np
from deepinv.models.diffunet import DiffUNet
import os

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set hyperparameters
    batch_size = 32
    num_epochs = 100
    lr = 1e-4
    image_size = 32  # MNIST is 28x28, but we'll resize to 32x32 for the model

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Limit the dataset to 100 images
    num_img = 1000
    indices = torch.arange(num_img)
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Initialize model
    model = DiffUNet(
        in_channels=1, out_channels=1, pretrained=None  # MNIST is grayscale
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define DDPM constants
    timesteps = 1000  # Total number of diffusion steps
    beta_start = 1e-4  # Starting value for noise schedule
    beta_end = 0.02  # Ending value for noise schedule

    # Linear noise schedule
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]]
    )
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # Training loop
    # Create a list to store all losses
    all_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            # Sample random timesteps
            t = torch.randint(0, timesteps, (images.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(images)

            # Apply forward diffusion process at timestep t
            noised_images = (
                sqrt_alphas_cumprod[t, None, None, None] * images
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )

            # Predict noise
            noise_pred = model(noised_images, t, type_t="timestep")

            # Calculate loss (the model predicts the noise that was added)
            loss = nn.MSELoss()(noise_pred, noise)

            loss.backward()
            optimizer.step()

            # Save the loss value
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        # Inference on random timesteps for visualizing training progress
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # Select a random batch from the training data for visualization
                sample_batch, _ = next(iter(train_loader))
                sample_batch = sample_batch.to(device)

                # Pick a single image from the batch
                sample_image = sample_batch[0:1]
                # Choose a single random timestep
                random_t = torch.randint(
                    1, 900, (1,)
                ).item()  # Choose a timestep where noise is visible
                t_tensor = torch.tensor([random_t], device=device)

                # Add noise according to the diffusion process
                noised_image = sqrt_alphas_cumprod[
                    random_t
                ] * sample_image + sqrt_one_minus_alphas_cumprod[
                    random_t
                ] * torch.randn_like(
                    sample_image
                )

                # Get model prediction (predicted noise)
                noise_pred = model(noised_image, t_tensor, type_t="timestep")

                # Denoise the image using the model prediction
                # x_0 = (x_t - sqrt(1-α_t) * predicted_noise) / sqrt(α_t)
                predicted_original = (
                    noised_image - sqrt_one_minus_alphas_cumprod[random_t] * noise_pred
                ) / sqrt_alphas_cumprod[random_t]

                # Clamp values to valid image range
                predicted_original = torch.clamp(predicted_original, -1, 1)

                # Save original image
                plt.figure(figsize=(5, 5))
                plt.imshow(sample_image.cpu().squeeze(0).squeeze(0), cmap="gray")
                plt.axis("off")
                plt.savefig(
                    f"./img/original/epoch_{epoch+1}_original.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()

                # Save noisy image
                plt.figure(figsize=(5, 5))
                plt.imshow(noised_image.cpu().squeeze(0).squeeze(0), cmap="gray")
                plt.axis("off")
                plt.savefig(
                    f"./img/noised/epoch_{epoch+1}_noised_t{random_t}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()

                # Save denoised image
                plt.figure(figsize=(5, 5))
                plt.imshow(predicted_original.cpu().squeeze(0).squeeze(0), cmap="gray")
                plt.axis("off")
                plt.savefig(
                    f"./img/denoised/epoch_{epoch+1}_denoised.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./weights/model_epoch_{epoch + 1}.pth")

    # Save final model
    torch.save(model.state_dict(), "./weights/model_final.pth")


if __name__ == "__main__":
    main()
