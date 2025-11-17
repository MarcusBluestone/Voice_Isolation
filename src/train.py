from data import CleanDataset
from data import NoiseGenerator, DataTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F

from autoencoder import AttnParams, CustomVAE

def train():

    """
    1. Sample chunk
    2. Add noise to chunk
    3. Convert into spectrogram of phase & amplitude
    4. Reshape to (256, 256)
    5. Input into Model
    6. Reshape output back into original size
    7. Loss = MSE btwn spectrograms
    8. Cann also convert spectrogram -> waveform -> audiofile if you want to hear what it actually sounds like  
    """

    # Setup Dataset
    dataset = CleanDataset(chunk_size = 30_000)
    train_loader = DataLoader(dataset, batch_size = 4, shuffle = True)

    # Setup Transformer & Augmenter
    data_transformer = DataTransformer()
    noise_generator = NoiseGenerator()

    # Setup Model & Optimizer
    attn_params = AttnParams(num_heads=4, window_size=None, use_rel_pos_bias=False, dim_head=64)
    model = CustomVAE(in_channels=2, spatial_dims=2, use_attn=True, attn_params=attn_params)

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
        noisy_waveform = noise_generator.add_gaussian(waveform)
        amp_noisy, phase_noisy, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
        
        # 3. Prepare Input to Model
        amp_inp, phase_inp = data_transformer.add_padding(amp_noisy), data_transformer.add_padding(phase_noisy)
        input = torch.stack((amp_inp, phase_inp), axis = 1)
        input = input.to(device)

        # 4. Run model & get loss
        output, z_mean, log_var = model(input)
        
        kl_per_elem = -0.5 * torch.sum(1 + log_var - z_mean.pow(2) - log_var.exp())
        # flatten all non-batch dims:
        kl_per_sample = kl_per_elem.view(kl_per_elem.size(0), -1).sum(dim=1)  # (B,)
        kl = kl_per_sample.mean()
        output = output[:, :, :W, :H]  # crop back to original size
        target = torch.stack((amp_clean, phase_clean), axis = 1).to(device)
        recon = F.mse_loss(output, target, reduction="none")
        recon = recon.view(recon.size(0), -1).mean(dim=1)         # per-sample mean
        recon_loss = recon.mean()  
        loss =  recon_loss + kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item())

        



if __name__ == '__main__':
    train()