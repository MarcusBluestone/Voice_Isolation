from tqdm import tqdm
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from data import CleanDataset
from data import NoiseGenerator, DataTransformer
from display_utils import plot_learning_curve

from autoencoder import AttnParams, CustomVAE

num_epochs = 50
dataset_size = 200
beta = 0
batch_size = 16
sigma_noise = 0.01

output_folder = Path('../outputs')
output_folder.mkdir(exist_ok=True)

def vae_loss(output, target, z_mean, log_var, beta):
    assert output.shape == target.shape

    # KL Loss
    kl = -0.5 * torch.mean(
        torch.sum(1 + log_var - z_mean.pow(2) - log_var.exp(), dim=1)
    )

    # Reconstruction loss
    recon_loss = F.mse_loss(output, target, reduction="none")
    recon_loss = recon_loss.view(recon_loss.size(0), -1).mean(dim=1) # per-sample mean
    recon_loss = recon_loss.mean() # per batch mean

    return recon_loss, kl, recon_loss + beta * kl

def evaluate(model, val_loader: DataLoader, data_transformer: DataTransformer, device, noise_generator: NoiseGenerator):
    loss_metrics = {
        'loss': 0, 
        'kl_loss': 0,
        'recon_loss': 0,
    }

    model.eval()
    for waveform, _ in tqdm(val_loader, "Evaluating"):
        # 1. Get Clean Chunk
        amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
        _, W,H = amp_clean.shape

        target = amp_clean.to(device).unsqueeze(1)

        # 2. Add Noise
        noisy_waveform = noise_generator.add_gaussian(waveform, sigma = sigma_noise)
        amp_noisy, _, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
        
        # 3. Prepare Input to Model
        input = data_transformer.add_padding(amp_noisy).unsqueeze(1).to(device)

        # 4. Run model & get loss
        output, z_mean, log_var = model(input)
        amp_recon = torch.tanh(output)[:, :, :W, :H]

        recon_loss, kl_loss, loss = vae_loss(amp_recon, target, z_mean, log_var, beta = beta)

        loss_metrics['loss'] += loss.cpu().detach().item()
        loss_metrics['kl_loss'] += kl_loss.cpu().detach().item()
        loss_metrics['recon_loss'] += recon_loss.cpu().detach().item()
    
    for loss_name in loss_metrics.keys():
        loss_metrics[loss_name] = (loss_metrics[loss_name] / len(val_loader))

    model.train()

    return loss_metrics['loss']


def train():
    # Setup Dataset
    train_dataset = CleanDataset(chunk_size = 30_000, count = dataset_size, split = 'train-clean-100')
    val_dataset = CleanDataset(chunk_size = 30_000, count = 10, split='dev-clean')

    train_loader = DataLoader(train_dataset, batch_size = batch_size,shuffle = False)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Setup Transformer & Augmenter
    data_transformer = DataTransformer()
    noise_generator = NoiseGenerator()

    # Setup Model & Optimizer
    attn_params = AttnParams(num_heads=4, window_size=None, use_rel_pos_bias=False, dim_head=64)
    model = CustomVAE(in_channels=1, spatial_dims=2, use_attn=False, vae_latent_channels=16,
                      attn_params=attn_params, vae_use_log_var = True, beta = beta, dropout_prob=0, blocks_down=(1,),
                      blocks_up = [])
    
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
    per_epoch_loss = {
        'loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'val_loss': []
    }

    # Perform the Training
    model.train()
    for epoch in range(1, num_epochs+1):
        loss_dict = { 
            'loss': 0,
            'recon_loss': 0,
            'kl_loss': 0,
        }

        for waveform, _ in tqdm(train_loader, f"Training Epoch #{epoch}:"):
            # 1. Get Clean Chunk
            amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
            _, W,H = amp_clean.shape

            target = amp_clean.to(device).unsqueeze(1)

            # 2. Add Noise
            noisy_waveform = noise_generator.add_gaussian(waveform, sigma = sigma_noise)
            amp_noisy, _, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
            
            # 3. Prepare Input to Model
            input = data_transformer.add_padding(amp_noisy).unsqueeze(1).to(device)

            # 4. Run model & get loss
            output, z_mean, log_var = model(input)
            amp_recon = torch.tanh(output)[:, :, :W, :H]

            recon_loss, kl_loss, loss = vae_loss(amp_recon, target, z_mean, log_var, beta = beta)

            loss_dict['loss'] += loss.cpu().detach()
            loss_dict['kl_loss'] += kl_loss.cpu().detach()
            loss_dict['recon_loss'] += recon_loss.cpu().detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for loss_name in loss_dict.keys():
            per_epoch_loss[loss_name].append(loss_dict[loss_name] / len(train_loader))
        per_epoch_loss['val_loss'].append(evaluate(model, val_loader, data_transformer, device, noise_generator))

        plot_learning_curve(per_epoch_loss, Path(output_folder /'lc.png'))
        pd.DataFrame(per_epoch_loss).to_csv(output_folder / 'epoch_metrics.csv')

    torch.save(model.state_dict(), output_folder / "model.pth")
    print(f"Saved Model to {output_folder}")

        



if __name__ == '__main__':
    train()