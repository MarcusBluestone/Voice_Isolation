from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader

from datasets import load_dataset
import torch
import torchaudio
import torch.nn.functional as F

import numpy as np
import os
import matplotlib.pyplot as plt
class CleanDatasetIterable(IterableDataset):
    """
    Streams the Data from Hugging Face. This avoids memory overhead
    of downloading the entire dataset.

    Two Problems:
    1. The program will not exit at the end. No bugs; just won't ever quit...
    2. No random access. 
    
    """
    def __init__(self, split: str = "train.100", chunk_size: int = 50_000):
        self.dataset = load_dataset("librispeech_asr", "clean", split=split, streaming=True)
        self.chunk_size = chunk_size

    def __iter__(self):
        for data in self.dataset:
            waveform = torch.tensor(data["audio"]["array"])
            sample_rate = data["audio"]["sampling_rate"]

            yield get_chunk(waveform, self.chunk_size), torch.tensor([sample_rate])

class CleanDataset(Dataset):
    """
    Downloads the Data from TorchAudio.Datasets. 
    
    - ~5 GB overhead for train
    - ~2 GB overhead for validate
    
    """
    def __init__(self, url="train-clean-100", chunk_size: int = 50_000):
        # (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root = './voice_data',
            url = url,
            download = True,
        )

        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate = self.dataset[idx][:2]

        return (get_chunk(waveform, self.chunk_size), torch.tensor([sample_rate]))

def get_chunk(waveform: torch.Tensor, chunk_size: int):
    """
    Takes in a 1D waveform and chunks it randomly w/ fixed size
    """
    idx = np.random.randint(0, len(waveform) - chunk_size)

    return waveform[idx:idx + chunk_size]

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
    Transform the Representation of the signal. Works on BATCHED inputs! If a vector is passed,
    it is unsqueezed first

    1. Waveform & Sample Rate  : (batch_size x num_samples) & (batch_size x 1)
    2. Spectrogram             : (batch_size x output_size x output_size)
    3. Image                   : .png of amplitude/phase for each elmnt in batch
    4. Audio File              : .wav file for each elmnt in batch

    a) waveform_to_audio: 1 -> 4
    b) waveform_to_spectrogram: 1 -> 2 
        - also returns min/max amp_db of original
    c) save_spectrogram: 2 -> 3
    d) spectrogram_to_waveform: 2 -> 1 
        - only gives waveform, not sample rate
        - request the min/max of amp_db of the original

    e) add_padding: (B, W, H) --> (B, 256, 256)

    """

    def __init__(self):

        # Values for Spectrogram creation
        self.nfft = 510
        self.hop_length = 256

    def waveform_to_audio(self, waveform: torch.Tensor, sample_rate: torch.Tensor, fname: str = 'output', max_save: int = 5): 
        # Unsqueeze if unbatched
        if waveform.ndim == 1:
            waveform = torch.unsqueeze(waveform, 0)
        if sample_rate.ndim == 1:
            sample_rate = torch.unsqueeze(sample_rate, 0)        

        assert waveform.shape[0] == sample_rate.shape[0] 

        num_samples = min(waveform.shape[0], max_save)

        for i in range(num_samples):
            torchaudio.save(f"{fname}{i}.wav", waveform[i, :], sample_rate[i, :])

    def _normalize_amplitude(self, amp_db):
        # amp_db: (B, F, T)
        min_db = amp_db.amin(dim=(1,2), keepdim=True)
        max_db = amp_db.amax(dim=(1,2), keepdim=True)

        amp_norm = (amp_db - min_db) / (max_db - min_db + 1e-8)
        return amp_norm, min_db, max_db

    def _denormalize_amplitude(self, amp_norm, min_db, max_db):
        return amp_norm * (max_db - min_db) + min_db

    def waveform_to_spectrogram(self, waveform: torch.Tensor):
        
        n_fft = self.nfft
        window = torch.hann_window(n_fft, device=waveform.device)  # Hann window

        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True
        )
        
        # 1. Calculate Amplitude
        magnitude = stft.abs()
        power = magnitude ** 2
        amplitude = torchaudio.transforms.AmplitudeToDB(stype='power', top_db = 80)(power) # 10 * log_10(power)
        amplitude, min_db, max_db = self._normalize_amplitude(amplitude)

        # 2. Calculate Phase
        phase = torch.angle(stft)

        # Phase: scale to [-1,1]
        phase = phase / torch.pi

        return amplitude, phase, (min_db, max_db)
    
    def spectrogram_to_waveform(self, amplitude: torch.Tensor, phase: torch.Tensor, min_db: torch.Tensor, max_db: torch.Tensor):
        """
        Invert everything from the function above
        """
        amplitude = self._denormalize_amplitude(amplitude, min_db, max_db)

        # convert from dB to linear power
        power = 10 ** (amplitude / 10) 
        magnitude = torch.sqrt(power)

        # phase: [-1,1] → [-pi, pi]
        phase = phase * torch.pi

        stft_recon = magnitude * torch.exp(1j * phase)  # complex tensor

        win_length = self.nfft
        window = torch.hann_window(self.nfft, device=stft_recon.device)

        # If batched, loop over batch
        waveforms = []
        for i in range(stft_recon.shape[0]):
            wav = torch.istft(
                stft_recon[i],         # shape: (freq_bins, time_frames)
                n_fft=self.nfft,
                hop_length= self.hop_length,
                win_length=win_length,
                window=window
            )
            waveforms.append(wav)

        waveforms = torch.stack(waveforms)  # (batch, num_samples)

        return waveforms
    
    def save_spectrogram(self, amp: torch.Tensor, phase: torch.Tensor, out_dir: str = 'outputs', max_save: int = 5):
        """
        Saves amplitude and phase spectrograms for a batch of waveforms.
        Each spectrogram gets its own file with a colorbar.

        Args:
            amp: (B, H, W) amplitude spectrograms, values in [0,1]
            phase: (B, H, W) phase spectrograms, values in [-1,1]
            out_dir: directory to save images
        """
        os.makedirs(out_dir, exist_ok=True)
        batch_size = amp.shape[0]

        for idx in range(min(max_save, batch_size)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # --- Amplitude ---
            im0 = axes[0].imshow(amp[idx].cpu().numpy(), cmap='magma', origin='lower')
            axes[0].set_title('Amplitude')
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0])

            # --- Phase ---
            im1 = axes[1].imshow(phase[idx].cpu().numpy(), cmap='twilight', vmin=-1.0, vmax=1.0, origin='lower')
            axes[1].set_title('Phase')
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1])

            plt.tight_layout()
            plt.savefig(f'{out_dir}/spec{idx}.png', dpi=150)
            plt.close()

    def add_padding(self, input: torch.Tensor, size: int = 256):
        """
        (B, W, H) -> (B, size, size) by padding on the right w/ 0's
        """
        B, W, H = input.shape
        
        pad_W = size - W
        pad_H = size - H

        # (pad_H_left, pad_H_right, pad_W_left, pad_W_right)
        return F.pad(input, (0, pad_H, 0, pad_W))

if __name__ == '__main__':
    dataset = CleanDatasetIterable(chunk_size = 50_000)
    td = TransformData()

    dataloader = DataLoader(dataset, batch_size=4)

    wave, rate = (next(iter(dataloader)))
    print(wave.shape)
    print(rate.shape)

    td.waveform_to_audio(wave, rate, fname = 'outputs/original', max_save = 3)

    amp, phase, (min_db, max_db) = td.waveform_to_spectrogram(wave)
    td.save_spectrogram(amp, phase)

    print(amp.shape)
    print(phase.shape)

    amp_res, phase_res = td.add_padding(amp), td.add_padding(phase)
    print(amp_res.shape)
    print(phase_res.shape)

    waveforms_reconstr = td.spectrogram_to_waveform(amp, phase, min_db, max_db)
    td.waveform_to_audio(waveforms_reconstr, rate, fname = 'outputs/reconstr', max_save = 3)





