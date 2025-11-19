from data import DataLoader, DataTransformer, NoiseGenerator
from autoencoder import autoencoder_loss
from tqdm import tqdm
import torch

def evaluate(model: torch.nn.Module, 
             val_loader: DataLoader, 
             data_transformer: DataTransformer, 
             device, 
             noise_generator: NoiseGenerator, 
             sigma_noise: float):
    
    loss_metrics = {
        'loss': 0, 
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
        output = model(input)
        amp_recon = output[:, :, :W, :H]

        loss = autoencoder_loss(amp_recon, target)

        loss_metrics['loss'] += loss.cpu().detach().item()

    for loss_name in loss_metrics.keys():
        loss_metrics[loss_name] = (loss_metrics[loss_name] / len(val_loader))

    model.train()

    return loss_metrics['loss']
