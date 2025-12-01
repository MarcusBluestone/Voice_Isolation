from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os

import torch
from torch.utils.data import DataLoader
from torch import optim

from data import CleanDataset
from data import NoiseGenerator, DataTransformer
from display_utils import plot_learning_curve
from evaluate import evaluate

# from vae import AttnParams, CustomVAE, vae_loss
from autoencoder import UNet, autoencoder_loss


def train(params: dict, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)

    # Read Params
    num_epochs = params['num_epochs']
    dataset_size = params['dataset_size']
    validation_size = params['validation_size']
    batch_size = params['batch_size']
    model_criteria = params['model_criteria']
    noise_type = params['noise_type']
    gauss_scale = params['gauss_scale']
    env_scale = params['env_scale']
    env_noise_type = params['env_type']


    # Setup Dataset
    train_dataset = CleanDataset(chunk_size = 30_000, count = dataset_size, split = 'train-clean-100')
    val_dataset = CleanDataset(chunk_size = 30_000, count = validation_size, split='dev-clean')

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
                noise_function = lambda x : noise_generator.add_gaussian(x, sigma = gauss_scale)  # noqa: E731
            elif noise_type == "E":
                noise_function = lambda x : noise_generator.add_environment(x, scale = env_scale, category_num = env_noise_type) # noqa: E731
            else:
                raise ValueError(f"Unknown noise type specified: {noise_type}")
            
            noisy_waveform = noise_function(waveform)
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
        per_epoch_loss['val_loss'].append(evaluate(model, val_loader, data_transformer, device, 
                                                   noise_fxn = noise_function))

        plot_learning_curve(per_epoch_loss, Path(out_dir /'lc.png'))
        pd.DataFrame(per_epoch_loss).to_csv(out_dir / 'epoch_metrics.csv')

        if per_epoch_loss[model_criteria][-1] <= np.min(per_epoch_loss[model_criteria]):
            torch.save(model.state_dict(), out_dir / "model.pth")
            print(f"Saved Model to {out_dir}")

def parse_params(param_dir: Path):
    param_runs = []

    for file_path in param_dir.iterdir():

        if file_path.is_file():
            print(f"Reading {file_path.name}") 

        with open(file_path, 'r') as f:
            param_runs.append(json.load(f))

    return param_runs

if __name__ == '__main__':
    param_dir = Path('../params')
    param_runs = parse_params(param_dir)

    for i, params in enumerate(param_runs):
        train(params, out_dir = Path(f'../outputs/G{i}'))