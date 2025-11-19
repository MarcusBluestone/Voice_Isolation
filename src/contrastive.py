from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np

from data import CleanDataset
from data_contrastive import AugmentedDataset
from data import NoiseGenerator, DataTransformer
from display_utils import plot_learning_curve

from vae import AttnParams, CustomVAE

from train import vae_loss, train

num_epochs = 200
dataset_size = 1
beta = 0
batch_size = 16
sigma_noise = 0.01

output_folder = Path('../outputs')
output_folder.mkdir(exist_ok=True)

def train_contrastive(autoencoder_model, latent_dim, Dataset, batch_size, epochs, device, tau: float = .07): 
    """
    Trains an encoder using contrastive learning of where clean voice signals are mapped close to
    those on which noise has been added, and far from other signals in the batch. Then trains the complete
    encoder-decoder model to reconstruct the clean signal from the latent representation.
    """ 
    
    # Setup Dataset
    dataset = AugmentedDataset(chunk_size = 30_000, count = dataset_size)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    # Setup Transformer & Augmenter
    data_transformer = DataTransformer()
    noise_generator = NoiseGenerator()

    # Setup Model & Optimizer
    attn_params = AttnParams(num_heads=4, window_size=None, use_rel_pos_bias=False, dim_head=64)
    model = CustomVAE(in_channels=1, spatial_dims=2, use_attn=False, vae_latent_channels=16,
                      attn_params=attn_params, vae_use_log_var = True, beta = beta, dropout_prob=0, blocks_down=(1,),
                      blocks_up = [])
    encoder = model.encode
    decoder = model.decode

    optim1 = torch.optim.Adam(list(model.parameters()), lr=2e-4)
    # optim2 = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-4)

    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    for epoch in range(epochs):
        for batch1, batch2 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # For each i, pair
            #   Positive pairs are (batch1[i], batch2[i])
            #   Negative pairs are (batch1[i], batch2[j]), j != i.
            amp1, _, _ = data_transformer.waveform_to_spectrogram(batch1)
            amp2, _, _ = data_transformer.waveform_to_spectrogram(batch2)
            input1 = data_transformer.add_padding(amp1).unsqueeze(0).to(device)
            input2 = data_transformer.add_padding(amp2).unsqueeze(0).to(device)
            # Contrastive training step
            optim1.zero_grad()
            # Run encoder on both inputs and get contrastive loss
            z_mean1, z_sigma1, eps1, logvar1 = encoder(input1)
            z_mean2, z_sigma2, eps2, logvar2 = encoder(input2)

            logits = (z_mean1 @ z_mean2.T)/tau
            criteria = nn.CrossEntropyLoss()
            loss1 = criteria(logits, torch.arange(logits.shape[0]).to(device))
            loss1.backward()
            optim1.step()

            # # Reconstruction training step
            # optim2.zero_grad()
            # enc_out_clean = enc(clean_batch)
            # dec_out = dec(enc_out_clean)
            # loss2 = F.mse_loss(dec_out, clean_batch)
            # loss2.backward()
            # optim2.step()

        print(f'[Contrastive] epoch {epoch: 4d}   Contrastive loss = {loss1.item():.4g}')

    # return enc, dec 