from torch.utils.data import Dataset

class CleanDataset(Dataset):
    """
    Dataset of clean audio signals  
    """
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


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
    """

    pass
