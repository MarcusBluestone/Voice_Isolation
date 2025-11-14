import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from src.data import CleanDataset, CleanDatasetIterable
from src.data import NoiseGenerator, DataTransformer

def train_contrastive(Encoder, Decoder, latent_dim, Dataset, batch_size, epochs, device, tau: float = .07): 
    """
    Trains an encoder using contrastive learning of where clean voice signals are mapped close to
    those on which noise has been added, and far from other signals in the batch. Then trains the complete
    encoder-decoder model to reconstruct the clean signal from the latent representation.
    """ 
    
    enc = Encoder(latent_dim).to(device)
    dec = Decoder(latent_dim).to(device)

    optim1 = torch.optim.Adam(list(enc.parameters()), lr=2e-4)
    optim2 = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-4)

    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    for epoch in range(epochs):
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            clean_batch = batch['clean'].to(device)
            noisy_batch = batch['noisy'].to(device)

            # Contrastive training step
            optim1.zero_grad()
            enc_out_clean = enc(clean_batch)
            enc_out_noisy = enc(noisy_batch)
            logits = (enc_out_clean @ enc_out_noisy.T)/tau
            criteria = nn.CrossEntropyLoss()
            loss1 = criteria(logits, torch.arange(logits.shape[0]).to(device))
            # loss1 = contrastive_loss(enc_out_clean, enc_out_noisy)
            loss1.backward()
            optim1.step()

            # Reconstruction training step
            optim2.zero_grad()
            enc_out_clean = enc(clean_batch)
            dec_out = dec(enc_out_clean)
            loss2 = F.mse_loss(dec_out, clean_batch)
            loss2.backward()
            optim2.step()

        print(f'[Contrastive] epoch {epoch: 4d}   Contrastive loss = {loss1.item():.4g}  Reconstruction loss = {loss2.item():.4g}')

    return enc, dec