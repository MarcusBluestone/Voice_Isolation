from data import DataLoader, DataTransformer
from tqdm import tqdm
import torch
from info_nce_loss import info_nce_loss

def evaluate_contrastive(model: torch.nn.Module, 
                         val_loader: DataLoader, 
                         data_transformer: DataTransformer, 
                         device, 
                         noise_fxn: callable,
                         tau: float = 0.07):
    """
    Evaluates the contrastive model on validation data.
    
    Args:
        model: The encoder-decoder model to evaluate
        val_loader: DataLoader for validation data
        data_transformer: DataTransformer for converting waveforms to spectrograms
        device: Device to run evaluation on (cpu/cuda/mps)
        noise_fxn: Function to add noise to waveforms
        tau: Temperature parameter for InfoNCE loss
    
    Returns:
        float: Average contrastive loss over the validation set
    """
    
    loss_metrics = {
        'contrastive_loss': 0, 
    }

    model.eval()
    encoder = model.encoder.to(device)
    
    with torch.no_grad():
        for waveform, _ in tqdm(val_loader, "Evaluating Contrastive"):
            # 1. Get Clean Chunk
            amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
            
            # 2. Add Noise
            noisy_waveform = noise_fxn(waveform)
            amp_noisy, _, _ = data_transformer.waveform_to_spectrogram(noisy_waveform)
            
            # 3. Prepare Input to Model
            input_clean = data_transformer.add_padding(amp_clean).unsqueeze(1).to(device)
            input_noisy = data_transformer.add_padding(amp_noisy).unsqueeze(1).to(device)

            # 4. Run encoder & get contrastive loss
            enc_clean, _ = encoder(input_clean)
            enc_noisy, _ = encoder(input_noisy)

            contrastive_loss = info_nce_loss(enc_clean, enc_noisy, tau=tau)

            loss_metrics['contrastive_loss'] += contrastive_loss.cpu().detach().item()

    # Calculate average loss
    avg_contrastive_loss = loss_metrics['contrastive_loss'] / len(val_loader)

    model.train()

    return avg_contrastive_loss
