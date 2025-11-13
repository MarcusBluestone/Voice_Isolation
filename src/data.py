from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset
import torch
import torchaudio

class CleanDatasetIterable(IterableDataset):
    def __init__(self, split="train.100", max_samples=5):
        self.dataset = load_dataset("librispeech_asr", "clean", split=split, streaming=True)

    def __iter__(self):
        for data in self.dataset:
            waveform = torch.tensor(data["audio"]["array"])
            sample_rate = data["audio"]["sampling_rate"]
            yield waveform, sample_rate

class CleanDataset(Dataset):
    def __init__(self, split="train.100"):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root = './voice_data',
            url = 'train-clean-100',
            download = True,
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]

        waveform = torch.tensor(data["audio"]["array"])
        sample_rate = data["audio"]["sampling_rate"]

        return waveform, sample_rate

class NoiseGenerator:
    """
    Contains a set of methods for adding noise to clean signals

    These methods are called in the train script
    """

    def __init__(self):
        # Download all the enviornment noise options? 
        pass

    def add_gaussian(self, amplitude: float):
        pass

    def add_environment(self, category: str):
        pass

    
class TransformData:
    """
    Transform the Representation of the signal
    - Wavelet
    - Spectrogram
    - Work on batches
    """

    pass

if __name__ == '__main__':
    dataset = CleanDataset()
    for i, (wave, rate) in enumerate(dataset):
        print(wave.shape, rate)
        torchaudio.save(f"output{i}.wav", wave, rate)


