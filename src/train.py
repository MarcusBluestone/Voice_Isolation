from data import CleanDataset
from data import NoiseGenerator, DataTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F

from autoencoder import AttnParams, CustomVAE

def train():
    # Setup Dataset
    dataset = CleanDataset(chunk_size = 30_000)
    train_loader = DataLoader(dataset, batch_size = 4, shuffle = True)

    # Setup Transformer & Augmenter
    data_transformer = DataTransformer()
    noise_generator = NoiseGenerator()

    # Setup Model & Optimizer
    attn_params = AttnParams(num_heads=4, window_size=None, use_rel_pos_bias=False, dim_head=64)
    model = CustomVAE(in_channels=2, spatial_dims=2, use_attn=True, attn_params=attn_params, vae_use_log_var = True)

    if torch.backends.mps.is_available():  # Apple Silicon GPU
        device = 'mps'
    elif torch.cuda.is_available():        # Nvidia GPU
        device = 'cuda'
    else:
        device = 'cpu'

    model = model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    # Perform the Training
    model.train()
    for waveform, _ in tqdm(train_loader, "Training:"):
        # 1. Get Clean Chunk
        amp_clean, phase_clean, _ = data_transformer.waveform_to_spectrogram(waveform)
        _, W,H = amp_clean.shape

        # 2. Add Noise
        noisy_waveform = noise_generator.add_gaussian(waveform, sigma = .01)
        amp_noisy, phase_noisy, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
        
        # 3. Prepare Input to Model
        amp_inp, phase_inp = data_transformer.add_padding(amp_noisy), data_transformer.add_padding(phase_noisy)
        input = torch.stack((amp_inp, phase_inp), axis = 1)
        input = input.to(device)

        # 4. Run model & get loss
        output, z_mean, log_var = model(input)

        # KL Loss
        kl = -0.5 * torch.sum(1 + log_var - z_mean.pow(2) - log_var.exp(), dim = 1)
        kl = kl.mean()

        # Reconstruction loss
        output = output[:, :, :W, :H]  # crop back to original size
        target = torch.stack((amp_clean, phase_clean), axis = 1).to(device)

        recon_loss = F.mse_loss(output, target, reduction="none")
        recon_loss = recon_loss.view(recon_loss.size(0), -1).mean(dim=1) # per-sample mean
        recon_loss = recon_loss.mean() # average across batch

        loss =  recon_loss + kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "./outputs/model.pth")

        



if __name__ == '__main__':
    train()