from data import DataLoader, DataTransformer
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn

def info_nce_loss(embeddings1, embeddings2, tau: float = .07):
    """
    Computes the InfoNCE loss between two sets of embeddings.
    embeddings1: Tensor of shape (batch_size, latent_dim)
    embeddings2: Tensor of shape (batch_size, latent_dim)
    tau: temperature parameter
    """
    batch_size = embeddings1.shape[0]
    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, dim=1).flatten(start_dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1).flatten(start_dim=1)

    z = torch.cat([embeddings1, embeddings2], dim=0)  # (2N, D)
    # Compute similarity matrix
    logits = (z @ z.T) / tau  # shape: (batch_size, batch_size)
    # mask self-similarity
    mask = torch.eye(logits.size(0), dtype=torch.bool).to(logits.device)
    logits = logits.masked_fill(mask, float('-inf'))

    # Create labels
    labels = torch.arange(2*batch_size).to(z.device)
    labels = (labels + batch_size) % (2*batch_size) # positive pairs are the other augmentation N away

    # Compute cross-entropy loss
    criteria = nn.CrossEntropyLoss()
    loss = criteria(logits, labels)

    return loss


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

    model.eval()  # evaluation mode
    
    encoder = model.encoder_contrastive
    
    with torch.no_grad():
        for waveform, _ in tqdm(val_loader, "Evaluating Contrastive"):
            # 1. Get Clean Chunk
            amp_clean, _, _ = data_transformer.waveform_to_spectrogram(waveform)
            
            # 2. Add Noise
            # noisy_waveform = waveform
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
    avg_contrastive_loss = (loss_metrics['contrastive_loss'] / len(val_loader))
    model.train()

    return avg_contrastive_loss, model.encoder_contrastive.parameters()
