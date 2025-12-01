import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(embeddings1, embeddings2, tau: float = .07):
    """
    Computes the InfoNCE loss between two sets of embeddings.
    embeddings1: Tensor of shape (batch_size, latent_dim_triplet)
    embeddings2: Tensor of shape (batch_size, latent_dim_triplet)


    At ith batch:
    - embeddings1[i] contains clean latent space vector
    - embeddings2[i] contains noisy latent space vector
    tau: temperature parameter

    """
    batch_size = embeddings1.shape[0]
    
    # Normalize embeddings
    embeddings1 = embeddings1.flatten(start_dim=1) # (B, L)
    embeddings2 = embeddings2.flatten(start_dim=1) # (B, L)

    embeddings1 = F.normalize(embeddings1, dim=1) #(B, L)
    embeddings2 = F.normalize(embeddings2, dim=1) #(B, L)


    z = torch.cat([embeddings1, embeddings2], dim=0)  # (2B, L)
    # Compute similarity matrix
    logits = (z @ z.T) / tau  # shape: (2B, 2B) (i,j) = sim(z_i, z_j)
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