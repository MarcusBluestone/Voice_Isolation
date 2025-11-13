from src.data import CleanDataset, CleanDatasetIterable
from src.data import NoiseGenerator, DataTransformer

def train():

    """
    1. Sample chunk
    2. Add noise to chunk
    3. Convert into spectrogram of phase & amplitude
    4. Reshape to (256, 256)
    5. Input into Model
    6. Reshape output back into original size
    7. Loss = MSE btwn spectrograms
    8. Cann also convert spectrogram -> waveform -> audiofile if you want to hear what it actually sounds like  
    """
    pass

if __name__ == '__main__':
    train()