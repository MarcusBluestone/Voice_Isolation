from shutil import copy
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data import CleanDataset
from data import NoiseGenerator, DataTransformer
from display_utils import plot_learning_curve
from evaluate import evaluate
from evaluate_contr import evaluate_contrastive
from inspect_contrastive_model import inspect_bottleneck, ablation_test_bottleneck, inspect_decoder_weights

# from vae import AttnParams, CustomVAE, vae_loss
from autoencoder import autoencoder_loss, UNet_double

from info_nce_loss import info_nce_loss

# def train_contrastive(autoencoder_model, latent_dim, Dataset, batch_size, epochs, device, tau: float = .07): 
def train_contrastive(params: dict, 
                      out_dir: Path, 
                      tau: float = .07, 
                      lam: float = 1.0, 
                      include_reconstruction: bool = False, 
                      train_reconstruction: bool = True,
                      validate: bool = True):
    """
    Trains an encoder using contrastive learning of where clean voice signals are mapped close to
    those on which noise has been added, and far from other signals in the batch. Then trains the complete
    encoder-decoder model to reconstruct the clean signal from the latent representation.
    """ 
    os.makedirs(out_dir, exist_ok=True)

    # Read Params
    num_epochs = params['num_epochs_contrastive']
    num_epochs_reconstruction = params['num_epochs_reconstruction']
    dataset_size = params['dataset_size']
    batch_size = params['batch_size']
    noise_type = params['noise_type']
    gauss_scale = params['gauss_scale']
    env_scale = params['env_scale']
    validation_size = params['validation_size']

    # Setup Dataset
    clean_dataset = CleanDataset(chunk_size = 30_000, count = dataset_size)
    val_dataset = CleanDataset(chunk_size = 30_000, count = validation_size, split='dev-clean')
    train_loader = DataLoader(clean_dataset, batch_size = batch_size, shuffle = False,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, drop_last=True)

    # Setup Transformer & Augmenter
    data_transformer = DataTransformer()
    noise_generator = NoiseGenerator()

    # Setup Model & Optimizer
    model = UNet_double(input_channels=1, base_filters=16, final_activation='tanh')
    encoder = model.encoder_contrastive
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        device = 'mps'
    elif torch.cuda.is_available():        # Nvidia GPU
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)
    encoder = encoder.to(device)
    optim = torch.optim.Adam(list(model.parameters()), lr=1e-3) # 2e-4

    if noise_type == "G":
        noise_function = lambda x : noise_generator.add_gaussian(x, sigma = gauss_scale)  # noqa: E731
    elif noise_type == "E":
        noise_function = lambda x : noise_generator.add_environment(x, scale = env_scale) # noqa: E731
    else:
        raise ValueError(f"Unknown noise type specified: {noise_type}")
    
    # Metrics
    per_epoch_loss = {
        'train_loss': [],
        'val_loss': []
    }

    # Perform the Training
    model.train()
    for epoch in range(1, num_epochs+1):
        loss_dict = { 
            'train_loss': 0.0,
            'contrastive_loss': 0.0,
            'reconstruction_loss': 0.0,
        }
        for waveform , _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            # 1. Get Clean Chunk
            amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
            _, W,H = amp_clean.shape

            # 2. Add Noise
            # noisy_waveform = waveform
            noisy_waveform = noise_function(waveform)
            amp_noisy, _, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)

            # 3. Prepare Input to Model
            input_clean = data_transformer.add_padding(amp_clean).unsqueeze(1).to(device)
            input_noisy = data_transformer.add_padding(amp_noisy).unsqueeze(1).to(device)

            # 4. Run model & get loss
            enc_clean, _ = encoder(input_clean)
            enc_noisy, _ = encoder(input_noisy)

            contrastive_loss = info_nce_loss(enc_clean, enc_noisy, tau=tau)
            loss_dict['contrastive_loss'] += contrastive_loss.cpu().detach()
            loss = contrastive_loss

            if include_reconstruction:
                target = amp_clean.to(device).unsqueeze(1)
                output = model(input_noisy)
                reconstruction_loss = 100 * autoencoder_loss(output[:, :, :W, :H], target)
                loss_dict['reconstruction_loss'] += reconstruction_loss.cpu().detach()
                loss += lam * reconstruction_loss
            loss_dict['train_loss'] += loss.cpu().detach()

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f'[Contrastive] epoch {epoch: 4d}   Contrastive loss = {loss_dict["contrastive_loss"]:.4g}   Reconstruction loss = {loss_dict["reconstruction_loss"]:.4g}   Total loss = {loss_dict["train_loss"]:.4g}')

        per_epoch_loss["train_loss"].append((loss_dict["train_loss"] / len(train_loader)).item())
        if validate:
            val_loss = evaluate_contrastive(model, val_loader, data_transformer, device,
                                                    noise_fxn = noise_function)

            per_epoch_loss['val_loss'].append(val_loss)
            print(f'    Validation loss = {per_epoch_loss["val_loss"][-1]:.4g}')
        
        plot_learning_curve(per_epoch_loss, Path(out_dir /'contrastive_real_lc.png'))
        pd.DataFrame(per_epoch_loss).to_csv(out_dir / 'contrastive_epoch_metrics.csv')

    save_path = out_dir / 'contrastive_model_no_decoder.pth'
    torch.save(model.state_dict(), save_path)
    print(per_epoch_loss)
    plot_learning_curve(per_epoch_loss, Path(out_dir /'contrastive_lc.png'))

    # ------------- RECONSTRUCTION -----------------
    # If we didn't include reconstruction during contrastive training,
    # train both the encoder and decoder here for reconstruction
    # train the decoder separately for reconstruction
    per_epoch_loss2 = {
        'train_reconstruction_loss': [],
        'val_reconstruction_loss': [],
        'val_reconstruction_loss_noise': [],
    }
    if train_reconstruction and not include_reconstruction:
        print("\n[Reconstruction] Training decoder for reconstruction...")
        model.train()
        model.freeze_contrastive_encoder()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, num_epochs_reconstruction + 1):
            reconstruction_loss_total = 0
            for waveform, _ in tqdm(train_loader, desc=f"Reconstruction Epoch {epoch}/{num_epochs}"):
                # 1. Get Clean Chunk
                amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
                _, W, H = amp_clean.shape
                
                # 2. Add Noise
                noise_function = lambda x: noise_generator.add_gaussian(x, sigma=gauss_scale)  # noqa: E731
                noisy_waveform = noise_function(waveform)
                amp_noisy, _, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
                
                # 3. Prepare Input
                input_noisy = data_transformer.add_padding(amp_noisy).unsqueeze(1).to(device)
                target = amp_clean.to(device).unsqueeze(1)
                
                # Decode and compute reconstruction loss
                output = model(input_noisy)
                reconstruction_loss = autoencoder_loss(output[:, :, :W, :H], target)
                
                optim.zero_grad()
                reconstruction_loss.backward()
                optim.step()
                
                reconstruction_loss_total += reconstruction_loss.cpu().detach()
            
            print(f'[Reconstruction] epoch {epoch: 4d}   Reconstruction loss = {reconstruction_loss_total / len(train_loader):.4g}')
            per_epoch_loss2['train_reconstruction_loss'].append(reconstruction_loss_total / len(train_loader))
            if validate:
                val_recon_loss = evaluate(model, val_loader, data_transformer, device,
                                         noise_fxn = noise_function)
                val_recon_loss_noise = evaluate(model, val_loader, data_transformer, device,
                                         noise_fxn = noise_function, use_random_contrastive=True)
                per_epoch_loss2['val_reconstruction_loss'].append(val_recon_loss)
                per_epoch_loss2['val_reconstruction_loss_noise'].append(val_recon_loss_noise)
                print(f'    Validation reconstruction loss = {val_recon_loss:.4g}')
                print(f'    Validation reconstruction loss (noise) = {val_recon_loss_noise:.4g}')

            plot_learning_curve(per_epoch_loss2, Path(out_dir /'reconstruction_post_contrastive_real_lc.png'))
            pd.DataFrame(per_epoch_loss2).to_csv(out_dir / 'reconstruction_post_contrastive_epoch_metrics.csv')

        save_path = out_dir / 'contrastive_model_with_decoder.pth'
        torch.save(model.state_dict(), save_path)
        print(f"\nModel with decoder saved to {save_path}")
        return model
        
        

if __name__ == "__main__":
    params = {
        'num_epochs_contrastive': 20,
        'num_epochs_reconstruction': 20,
        'dataset_size': None,
        'validation_size': None,
        'batch_size': 128,
        'model_criteria': 'mse',
        'noise_type': 'G',
        'gauss_scale': 0.1,
        'env_scale': 1,
    }
    run_gaussian = [.01, .1, .3, .5]
    names = ['G0-01', 'G0-1', 'G0-3', 'G0-5']
    for scale, name in zip(run_gaussian, names):
        exp_params = copy.deepcopy(params)
        exp_params['gauss_scale'] = scale
        out_dir = Path(f'../outputs/contrastive/{name}')
        model = train_contrastive(exp_params, out_dir, tau=0.07, validate=True)

        if torch.backends.mps.is_available():  # Apple Silicon GPU
            device = 'mps'
        elif torch.cuda.is_available():        # Nvidia GPU
            device = 'cuda'
        else:
            device = 'cpu'
        model = model.to(device)
        clean_dataset = CleanDataset(chunk_size = 30_000, count = 50)
        train_loader = DataLoader(clean_dataset, batch_size = 64, shuffle = False, drop_last=True)

        file_out = out_dir / "contrastive_model_inspection.txt"
        inspect_bottleneck(model, train_loader, file_out, device=device)
        ablation_test_bottleneck(model, train_loader, DataTransformer(), file_out, device=device, num_batches=10)
        inspect_decoder_weights(model, file_out)