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

# from vae import AttnParams, CustomVAE, vae_loss
from autoencoder import autoencoder_loss, UNet_double

def inspect_bottleneck(model, data_loader, out_file, device='cuda'):
    """
    Inspect the average activation magnitude of the two encoder bottlenecks.

    This runs a few batches through the contrastive and regular encoders
    separately, computes the mean absolute value of their latent activations,
    and prints the results. Useful for checking whether the contrastive
    encoder is active or collapsed.

    Usage:
        model.eval()
        inspect_bottleneck(model, train_loader, device="cuda")

    Args:
        model: UNet_double model with encoder_contrastive and encoder_regular.
        data_loader: DataLoader yielding (waveform, label).
        device: "cuda", "mps", or "cpu".
    """
    model.eval()
    contrastive_mags = []
    regular_mags = []
    data_transformer = DataTransformer()

    with torch.no_grad():
        for waveform, _ in data_loader:
            # Get spectrogram
            amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
            x = data_transformer.add_padding(amp_clean).unsqueeze(1).to(device)

            # Forward through each encoder separately
            z_c, _ = model.encoder_contrastive(x)   # (B, Cc, H, W)
            z_r, _ = model.encoder_regular(x)       # (B, Cr, H, W)

            # Compute magnitude
            contrastive_mags.append(z_c.abs().mean().item())
            regular_mags.append(z_r.abs().mean().item())

            # Only need a few batches
            if len(contrastive_mags) > 30:
                break

    line_1 = f"Contrastive latent mean abs activation: {sum(contrastive_mags) / len(contrastive_mags)}"
    line_2 = f"Regular latent mean abs activation: {sum(regular_mags) / len(regular_mags)}"

    with open(out_file, "a") as f:
        f.write("\n".join([line_1, line_2]))

def ablation_test_bottleneck(model, data_loader, data_transformer, out_file, device="cuda", num_batches=5):
    """
    Perform a bottleneck ablation study by zeroing out encoder branches.

    For each batch, this function:
        1. Runs the full model (both bottlenecks).
        2. Zeros out contrastive features and recomputes output.
        3. Zeros out regular features and recomputes output.
        4. Measures how much each change affects the decoder output.

    This reveals which encoder branch the decoder relies on.

    Usage:
        model.eval()
        ablation_test_bottleneck(
            model,
            train_loader,
            DataTransformer(),
            device="cuda",
            num_batches=10
        )

    Args:
        model: UNet_double model with both encoders.
        data_loader: DataLoader yielding (waveform, label).
        data_transformer: Instance providing waveform_to_spectrogram() and padding.
        device: Device to run on.
        num_batches: How many batches to test.
    """
    model.eval()
    diffs_no_contrastive = []
    diffs_no_regular = []

    with torch.no_grad():
        for i, (waveform, _) in enumerate(data_loader):
            # 1. Get input spectrogram
            amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
            x = data_transformer.add_padding(amp_clean).unsqueeze(1).to(device)
            target = amp_clean.to(device).unsqueeze(1)

            # 2. Run both encoders
            z_c, _ = model.encoder_contrastive(x)   # (B, Cc, H', W')
            z_r, enc_feats = model.encoder_regular(x)  # (B, Cr, H', W')

            # 3. Full bottleneck
            bottleneck_full = torch.cat([z_c, z_r], dim=1)  # (B, Cc+Cr, H', W')
            out_full = model.decoder(bottleneck_full, enc_feats)

            # 4. Zero out contrastive part
            z_c_zero = torch.zeros_like(z_c)
            bottleneck_no_c = torch.cat([z_c_zero, z_r], dim=1)
            out_no_c = model.decoder(bottleneck_no_c, enc_feats)

            # 5. Zero out regular part
            z_r_zero = torch.zeros_like(z_r)
            bottleneck_no_r = torch.cat([z_c, z_r_zero], dim=1)
            out_no_r = model.decoder(bottleneck_no_r, enc_feats)

            # 6. Compare to full output (or to target)
            # here: difference vs full output
            diff_no_c = F.mse_loss(out_no_c, out_full).item()
            diff_no_r = F.mse_loss(out_no_r, out_full).item()

            diffs_no_contrastive.append(diff_no_c)
            diffs_no_regular.append(diff_no_r)

            if i + 1 >= num_batches:
                break

    line_1 = f"Mean MSE(out_no_contrastive, out_full): {sum(diffs_no_contrastive) / len(diffs_no_contrastive)}"
    line_2 = f"Mean MSE(out_no_regular,      out_full): {sum(diffs_no_regular) / len(diffs_no_regular)}"

    with open(out_file, "a") as f:
        f.write("\n".join([line_1, line_2]))

def inspect_decoder_weights(model, out_file):
    """
    Inspect how strongly the decoder weights connect to each bottleneck branch.

    Examines the first decoder layer that consumes the concatenated bottleneck,
    splits its input channels into contrastive vs regular groups, and reports
    their mean absolute weight values. Useful for determining whether the
    decoder actually uses the contrastive pathway.

    Usage:
        inspect_decoder_weights(model)

    Args:
        model: UNet_double with decoder.up6 as the first bottleneck-consuming layer.
    """
    # First layer that consumes the bottleneck:
    w = model.decoder.up6.weight    # shape: (out_ch, in_ch, kH, kW)

    in_ch = w.shape[0] # for convTranspose the in_channels are at dim 0
    Cc = model.encoder_contrastive.conv5.conv2.out_channels  # 16 * base_filters
    Cr = model.encoder_regular.conv5.conv2.out_channels      # same

    assert Cc + Cr == in_ch, f"Expected {Cc+Cr} in_channels, got {in_ch}"

    # Split along the in_channel dimension (dim=0)
    w_contrastive = w[:Cc].abs().mean().item()
    w_regular     = w[Cc:].abs().mean().item()

    line_1 = f"Decoder up6 | mean | contrastive input channels: {w_contrastive}"
    line_2 = f"Decoder up6 | mean | regular      input channels: {w_regular}"

    with open(out_file, "a") as f:
        f.write("\n".join([line_1, line_2]))

# model = UNet_double(input_channels=1, base_filters=16, final_activation='tanh')
# if torch.backends.mps.is_available():  # Apple Silicon GPU
#     device = 'mps'
# elif torch.cuda.is_available():        # Nvidia GPU
#     device = 'cuda'
# else:
#     device = 'cpu'
# model = model.to(device)
# model.load_state_dict(torch.load("/data/rbg/users/duitz/Voice_Isolation/outputs/contrastive/third_G1/contrastive_model_with_decoder.pth", map_location=device))


# clean_dataset = CleanDataset(chunk_size = 30_000, count = 50)
# train_loader = DataLoader(clean_dataset, batch_size = 16, shuffle = False,drop_last=True)
# inspect_bottleneck(model, train_loader, device=device)
# ablation_test_bottleneck(model, train_loader, DataTransformer(), device=device, num_batches=10)

# inspect_decoder_weights(model)
