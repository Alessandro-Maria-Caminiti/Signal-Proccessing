import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFourierLoss(nn.Module):
    def __init__(self, alpha_init=1.0, smoothing=0.01):
        """
        A dynamic loss that balances time-domain and frequency-domain MSE.

        Parameters:
        - alpha_init: Initial weighting for the frequency-domain loss.
        - smoothing: Exponential smoothing factor for updating alpha.
        """
        super(DynamicFourierLoss, self).__init__()
        self.alpha = torch.tensor(alpha_init, dtype=torch.float32)
        self.smoothing = smoothing

    def forward(self, pred_waveform, target_waveform):
        # Ensure alpha is on the correct device
        device = pred_waveform.device
        self.alpha = self.alpha.to(device)

        # Time domain loss (MSE)
        time_loss = F.mse_loss(pred_waveform, target_waveform)


        # Frequency domain loss (MSE of magnitude spectra)
        pred_mag = torch.abs(torch.fft.rfft(pred_waveform, dim=-1))
        target_mag = torch.abs(torch.fft.rfft(target_waveform, dim=-1))
        spectral_convergence = torch.norm(pred_mag - target_mag, p='fro') / torch.norm(target_mag, p='fro')
        freq_loss = 0.7 * F.l1_loss(pred_mag, target_mag) + 0.3 * spectral_convergence 

        # Update alpha using exponential smoothing (detach to avoid autograd)
        with torch.no_grad():
            loss_ratio = freq_loss / (time_loss + 1e-6)
            self.alpha = (1 - self.smoothing) * self.alpha + self.smoothing * loss_ratio.detach()

        # Combined dynamic loss
        total_loss = time_loss * (1 - self.alpha) + freq_loss * self.alpha
        return total_loss