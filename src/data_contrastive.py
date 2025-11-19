from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from datasets import load_dataset
import torch
import torchaudio
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# from data import CleanDataset
from data import NoiseGenerator, DataTransformer, get_chunk
gaussian_noise = NoiseGenerator().add_gaussian

class AugmentedDataset(Dataset):
    """
    Dataset that provides multiple augmentations of each audio sample for contrastive learning.

    Each item in the dataset returns `num_samples` augmented versions of the same audio chunk.

    TODO: allow multiple chunks from same waveform
        - Try different noise transforms
    
    """
    def __init__(self, split="train-clean-100", transform=gaussian_noise, num_samples: int = 1, chunk_size: int = 50_000, count: int | None = None):
        # (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root = '../voice_data',
            url = split,
            download = True,
        )
        if count is not None:
            self.dataset = Subset(self.dataset, indices= range(min(count, len(self.dataset))))

        if transform is None:
            transform = lambda x: x
        self.transform = transform
        self.num_samples = num_samples
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.dataset)
    
    def get_augs(self, idx, num_samples):
        waveform, sample_rate = self.dataset[idx][:2]
        waveform = waveform[0]  # Get the 1D tensor from the 2D tensor

        chunk = torch.tensor(get_chunk(waveform, self.chunk_size))

        return (chunk,) + tuple(self.transform(chunk) for _ in range(num_samples))
    
    def __getitem__(self, idx):
        return self.get_augs(idx, self.num_samples)