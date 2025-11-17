from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from data import CleanDataset
from data import NoiseGenerator, DataTransformer
from display_utils import plot_learning_curve

from autoencoder import AttnParams, CustomVAE

num_epochs = 30
dataset_size = 1000
beta = 0
batch_size = 16

output_folder = Path('../outputs')
output_folder.mkdir(exist_ok=True)

def vae_loss(output, target, z_mean, log_var, beta):
    assert output.shape == target.shape

    # KL Loss
    kl = -0.5 * torch.sum(1 + log_var - z_mean.pow(2) - log_var.exp(), dim=1)
    kl = kl.mean()

    # Reconstruction loss
    recon_loss = F.mse_loss(output, target, reduction="none")
    recon_loss = recon_loss.view(recon_loss.size(0), -1).mean(dim=1) # per-sample mean
    recon_loss = recon_loss.mean() # average across batch

    return recon_loss, kl, recon_loss + beta * kl

def train():
    # Setup Dataset
    dataset = CleanDataset(chunk_size = 30_000, count = dataset_size)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # Setup Transformer & Augmenter
    data_transformer = DataTransformer()
    noise_generator = NoiseGenerator()

    # Setup Model & Optimizer
    attn_params = AttnParams(num_heads=4, window_size=None, use_rel_pos_bias=False, dim_head=64)
    model = CustomVAE(in_channels=2, spatial_dims=2, use_attn=True,
                      attn_params=attn_params, vae_use_log_var = True, beta = beta)

    if torch.backends.mps.is_available():  # Apple Silicon GPU
        device = 'mps'
    elif torch.cuda.is_available():        # Nvidia GPU
        device = 'cuda'
    else:
        device = 'cpu'

    model = model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Metrics
    epoch_total_losses = []
    epoch_recon_losses = []
    epoch_kl_losses = []

    # Perform the Training
    model.train()
    for epoch in range(1, num_epochs+1):

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for waveform, _ in tqdm(train_loader, f"Training Epoch #{epoch}:"):
            # 1. Get Clean Chunk
            amp_clean, phase_clean, _ = data_transformer.waveform_to_spectrogram(waveform)
            _, W,H = amp_clean.shape

            target = torch.stack((amp_clean, phase_clean), axis = 1).to(device)
            
            # 2. Add Noise
            # noisy_waveform = noise_generator.add_gaussian(waveform, sigma = .01)
            noisy_waveform = waveform
            amp_noisy, phase_noisy, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
            
            # 3. Prepare Input to Model
            input = torch.stack((amp_noisy, phase_noisy), axis = 1)
            input = data_transformer.add_padding(input)
            input = input.to(device)

            # 4. Run model & get loss
            output, z_mean, log_var = model(input)

            recon_loss, kl_loss, loss = vae_loss(output[:, :, :W, :H], target, z_mean, log_var, beta = beta)

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_total_losses.append(total_loss / len(train_loader))
        epoch_kl_losses.append(total_kl_loss / len(train_loader))
        epoch_recon_losses.append(total_recon_loss / len(train_loader))

        print("Loss:", epoch_total_losses[-1])
        plot_learning_curve({
            'total_loss': epoch_total_losses,
            'kl_loss': epoch_kl_losses,
            'recon_loss': epoch_recon_losses
        }, Path(output_folder /'lc.png'))

    torch.save(model.state_dict(), output_folder / "model.pth")

        



if __name__ == '__main__':
    train()