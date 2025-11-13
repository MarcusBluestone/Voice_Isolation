from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader

from datasets import load_dataset
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

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
    def __init__(self, split="train.100"):
        self.dataset = load_dataset("librispeech_asr", "clean", split=split, streaming=True)

    def __iter__(self):
        for data in self.dataset:
            waveform = torch.tensor(data["audio"]["array"])
            sample_rate = data["audio"]["sampling_rate"]

            yield waveform, torch.tensor([sample_rate])

class CleanDataset(Dataset):
    """
    Downloads the Data from TorchAudio.Datasets. 
    
    - ~5 GB overhead for train
    - ~2 GB overhead for validate
    
    """
    def __init__(self, url="train-clean-100"):
        # (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root = './voice_data',
            url = url,
            download = True,
        )


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate = self.dataset[idx][:2]

        return (waveform, torch.tensor([sample_rate]))

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

    waveform_to_audio: 1 -> 4
    waveform_to_spectrogram: 1 -> 2
    save_spectrogram: 2 -> 3
    spectrogram_to_waveform: 2 -> 1 (only gives waveform, not sample rate)

    fix_waveform_length: should be used in DataLoader collate_fn
    """

    def __init__(self):

        # Values for Spectrogram creation
        self.nfft = 1024
        self.hop_length = 512

        self.amp_min = -80
        self.amp_max = 0
        
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
        power = stft.abs() ** 2
        # This does: 10 * log_10(power)
        amplitude = torchaudio.transforms.AmplitudeToDB(stype='power')(power) 

        # Amplitude: scale to [0,1]
        amplitude = torch.clamp(amplitude, min=self.amp_min, max=self.amp_max)
        amplitude = (amplitude - self.amp_min) / (-self.amp_min)

        # 2. Calculate Phase
        phase = torch.angle(stft)

        # Phase: scale to [-1,1]
        phase = phase / torch.pi

        return amplitude, phase
    
    def spectrogram_to_waveform(self, amplitude: torch.Tensor, phase: torch.Tensor):
        amplitude = amplitude * -self.amp_min + self.amp_min

        # convert from dB to linear power
        amplitude = 10 ** (amplitude / 10) 

        # phase: [-1,1] → [-pi, pi]
        phase = phase * torch.pi

        stft_recon = amplitude * torch.exp(1j * phase)  # complex tensor

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
    
    @staticmethod
    def fix_waveform_length(batch, desired_size = 220_000):
        """
        batch: list of tuples (waveform, sample_rate)
        Pads/Cuts all waveforms to a specified length
        """
        waveforms, sample_rates = zip(*batch)
 
        padded_waveforms = pad_sequence(waveforms + (torch.ones(desired_size),), batch_first = True) # expand if necessary
        return padded_waveforms[:-1, :desired_size], torch.tensor(sample_rates).unsqueeze(1)

    @staticmethod
    def save_spectrogram(amp: torch.Tensor, phase: torch.Tensor, out_dir: str = 'outputs', max_save: int = 5):
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
            im0 = axes[0].imshow(amp[idx].cpu().numpy(), cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
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

if __name__ == '__main__':
    dataset = CleanDatasetIterable()
    td = TransformData()

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=td.fix_waveform_length)

    wave, rate = (next(iter(dataloader)))
    print(wave.shape)
    print(rate.shape)

    td.waveform_to_audio(wave, rate, fname = 'outputs/output', max_save = 3)

    amp, phase = td.waveform_to_spectrogram(wave)
    td.save_spectrogram(amp, phase)

    print(amp.shape)
    print(phase.shape)

    waveforms_reconstr = td.spectrogram_to_waveform(amp, phase)
    td.waveform_to_audio(waveforms_reconstr, rate, fname = 'outputs/reconstr', max_save = 3)





