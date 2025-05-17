import torch


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