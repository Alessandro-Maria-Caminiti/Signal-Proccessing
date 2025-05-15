import torch
from pesq import pesq
import numpy as np

def compute_pesq(clean, denoised, sr):
    """
    Computes PESQ for each sample in batch.
    clean, denoised: tensors of shape [batch_size, num_samples]
    Returns: mean PESQ score
    """
    scores = []
    for c, d in zip(clean, denoised):
        try:
            c_np = c.squeeze().cpu().numpy()
            d_np = d.squeeze().cpu().numpy()
            score = pesq(sr, c_np, d_np, 'wb')  # 'wb' = wideband (16kHz)
            scores.append(score)
        except:
            continue  # Skip if PESQ fails due to NaNs or sample mismatch
    return np.mean(scores) if scores else 0.0


def compute_sdr(clean, denoised, eps=1e-8):
    """
    Computes SDR between clean and denoised audio signals.
    
    clean, denoised: tensors of shape [batch_size, num_samples]
    Returns: mean SDR over the batch
    """
    clean_energy = torch.sum(clean ** 2, dim=1) + eps
    noise = clean - denoised
    noise_energy = torch.sum(noise ** 2, dim=1) + eps
    sdr = 10 * torch.log10(clean_energy / noise_energy)
    return sdr.mean()
