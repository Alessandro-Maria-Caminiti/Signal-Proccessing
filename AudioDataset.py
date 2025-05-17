import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import random
import torch.nn.functional as F
from torchaudio.transforms import Resample, MelSpectrogram
import pandas as pd



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================
# 1. Custom Dataset Definition
# ============================
class AudioPairDataset(Dataset):
    def __init__(self, list_path, transform=None): 
        self.pathlist = list_path
        self.transform = transform

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        clean_path = self.pathlist[idx]
        clean_waveform, cleanwaveform_sr = torchaudio.load(clean_path)
        if self.transform:
            clean_waveform = self.transform(clean_waveform, cleanwaveform_sr)
        distorted_waveform = None
        if np.random.rand() < 0.5:
            distorted_waveform = add_distortion(clean_waveform)
            distorted_waveform = add_echo(distorted_waveform) 
        else:
            if np.random.rand() < 0.5:
                distorted_waveform = add_distortion(clean_waveform) 
            else:
                distorted_waveform = add_echo(clean_waveform)

        return distorted_waveform, clean_waveform








def collate_fn(batch):
    distorted, clean = zip(*batch)
    distorted = torch.stack([d.to(torch.float32).to(device) for d in distorted])
    clean = torch.stack([c.to(torch.float32).to(device) for c in clean])
    return distorted, clean



# ==============================
# 2. Subset Sampling & DataLoader
# ==============================
def create_dataloader(
    clean_dir,
    # subset_size=100,
    batch_size=32,
    transform=None,
    shuffle=True,
    num_workers=0
):
    # Initialize full dataset
    preprocessor = AudioPreprocessor(target_sr=16000, num_samples=32000)
    full_dataset = AudioPairDataset(clean_dir, transform=preprocessor)

    # # Randomly sample subset
    # indices = random.sample(range(len(full_dataset)), subset_size)
    # subset = Subset(full_dataset, indices)

    # Collate function to move to GPU

    # Create DataLoader
    loader = DataLoader(
        full_dataset,
        # subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device == "cpu" else False,
    )

    return loader











# ==============================
# 3. Audioprocessing & Feature Extraction
# ==============================
class AudioPreprocessor:
    """
    A class for preprocessing audio waveforms, including resampling, normalization, 
    padding/truncation, and conversion to Mel spectrograms.
    Attributes:
        target_sr (int): The target sampling rate for resampling the audio.
        num_samples (int): The fixed number of samples for the audio waveform after padding or truncation.
        resampler (Resample): A resampling module to convert the audio to the target sampling rate.
        melspec (MelSpectrogram): A module to convert the audio waveform to a Mel spectrogram.
    Methods:
        __call__(waveform, orig_sr):
            Processes the input waveform by performing the following steps:
            1. Resamples the waveform to the target sampling rate if it differs from the original sampling rate.
            2. Converts the waveform to mono by averaging across channels if it has multiple channels.
            3. Normalizes the waveform to the range [-1, 1].
            4. Pads or truncates the waveform to ensure it has a fixed number of samples.
            5. Converts the processed waveform to a Mel spectrogram representation.
    Args:
        waveform (Tensor): The input audio waveform as a PyTorch tensor of shape [channels, samples].
        orig_sr (int): The original sampling rate of the input waveform.
    Returns:
        Tensor: A Mel spectrogram representation of the processed waveform with shape [1, n_mels, time].
    """
    def __init__(self, target_sr, num_samples):
        self.target_sr = target_sr
        self.num_samples = num_samples
        self.melspec = MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )

    def __call__(self, waveform, orig_sr):
        # Resample
        if orig_sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.target_sr)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize to [-1, 1]
        waveform = waveform / waveform.abs().max()

        # Pad or truncate to fixed length
        if waveform.shape[1] < self.num_samples:
            pad_len = self.num_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.num_samples]
        

        # Convert to Mel Spectrogram
        mel = self.melspec(waveform)  # shape: [1, n_mels, time]
        mel = F.pad(mel, (0, 2))
        return mel









# ==============================
# 4. Distortion Function
# ==============================
def add_echo(waveform, delay=16000, decay=0.5):
    """
    Adds an echo effect to the waveform.
    Args:
        waveform (Tensor): The input audio waveform of shape [1, samples].
        delay (int): The delay in samples for the echo.
        decay (float): The decay factor for the echo amplitude.
    Returns:
        Tensor: The waveform with the echo effect applied.
    """
    echo_waveform = torch.zeros_like(waveform)
    if waveform.shape[1] > delay:
        echo_waveform[:, delay:] = waveform[:, :-delay] * decay
    return waveform + echo_waveform

def add_distortion(waveform, gain=10):
    """
    Adds distortion to the waveform by applying a gain and clipping.
    Args:
        waveform (Tensor): The input audio waveform of shape [1, samples].
        gain (float): The gain factor to amplify the waveform.
    Returns:
        Tensor: The waveform with distortion applied.
    """
    distorted_waveform = waveform * gain
    return torch.clamp(distorted_waveform, -1.0, 1.0)
