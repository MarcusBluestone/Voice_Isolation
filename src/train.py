from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import json

import torch
from torch.utils.data import DataLoader
from torch import optim

from data import CleanDataset
from data import NoiseGenerator, DataTransformer
from display_utils import plot_learning_curve
from evaluate import evaluate

# from vae import AttnParams, CustomVAE, vae_loss
from autoencoder import UNet, autoencoder_loss


output_folder = Path('../outputs')
output_folder.mkdir(exist_ok=True)

def train(params: dict):
    # Read Params
    num_epochs = params['num_epochs']
    dataset_size = params['dataset_size']
    batch_size = params['batch_size']
    sigma_noise = params['sigma_noise']
    model_criteria = params['model_criteria']
    noise_type = params['noise_type']

    # Setup Dataset
    train_dataset = CleanDataset(chunk_size = 30_000, count = dataset_size, split = 'train-clean-100')
    val_dataset = CleanDataset(chunk_size = 30_000, count = 10, split='dev-clean')

    train_loader = DataLoader(train_dataset, batch_size = batch_size,shuffle = False)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Setup Transformer & Augmenter
    data_transformer = DataTransformer()
    noise_generator = NoiseGenerator()

    # Setup Model & Optimizer
    model = UNet(input_channels = 1, final_activation='tanh')
    
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
        'val_loss': []
    }

    # Perform the Training
    model.train()
    for epoch in range(1, num_epochs+1):
        loss_dict = { 
            'loss': 0,
        }

        for waveform, _ in tqdm(train_loader, f"Training Epoch #{epoch}:"):
            # 1. Get Clean Chunk
            amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
            _, W,H = amp_clean.shape

            target = amp_clean.to(device).unsqueeze(1)

            # 2. Add Noise
            if noise_type == "G":
                noisy_waveform = noise_generator.add_gaussian(waveform, sigma = sigma_noise)
            elif noise_type == "E":
                noisy_waveform = noise_generator.add_environment(waveform, scale = 10)
            else:
                raise ValueError(f"Unknown noise type specified: {noise_type}")
            
            amp_noisy, _, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
            
            # 3. Prepare Input to Model
            input = data_transformer.add_padding(amp_noisy).unsqueeze(1).to(device)

            # 4. Run model & get loss
            output = model(input)
            amp_recon = output[:, :, :W, :H]

            # recon_loss, kl_loss, loss = vae_loss(amp_recon, target, z_mean, log_var, beta = beta)
            loss = autoencoder_loss(amp_recon, target)
            loss_dict['loss'] += loss.cpu().detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for loss_name in loss_dict.keys():
            per_epoch_loss[loss_name].append(loss_dict[loss_name] / len(train_loader))
        per_epoch_loss['val_loss'].append(evaluate(model, val_loader, data_transformer, device, noise_generator, sigma_noise))

        plot_learning_curve(per_epoch_loss, Path(output_folder /'lc.png'))
        pd.DataFrame(per_epoch_loss).to_csv(output_folder / 'epoch_metrics.csv')

        if per_epoch_loss[model_criteria][-1] <= np.min(per_epoch_loss[model_criteria]):
            torch.save(model.state_dict(), output_folder / "model.pth")
            print(f"Saved Model to {output_folder}")

def parse_params(param_path: Path):
    params = {}
    with open(param_path, 'r') as f:
        params = json.load(f)

    return params

if __name__ == '__main__':
    param_path = Path('../params/run1.json')
    params = parse_params(param_path)

    train(params)