import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a custom loss function class that inherits from nn.Module
class DynamicFourierLoss(nn.Module):
    def __init__(self, alpha_init=1.0, smoothing=0.01):
        """
        Initialize the DynamicFourierLoss class.

        Parameters:
        - alpha_init: Initial weight for the frequency domain loss (default is 1.0).
                      This determines the initial importance of the frequency loss relative to the time loss.
        - smoothing: A factor (between 0 and 1) that controls how quickly the alpha value adjusts.
                     Lower values mean slower adjustments, while higher values mean faster adjustments.
        """
        super(DynamicFourierLoss, self).__init__()  # Initialize the parent nn.Module class
        self.alpha = alpha_init  # Initialize the alpha value
        self.smoothing = smoothing  # Store the smoothing factor

    def forward(self, pred_waveform, target_waveform):
        """
        Compute the loss between the predicted and target waveforms.

        Parameters:
        - pred_waveform: The predicted waveform (tensor).
        - target_waveform: The target waveform (tensor).

        Returns:
        - total_loss: The combined loss (time domain + frequency domain) with dynamic weighting.
        """
        # 1. Time domain loss (mean squared error between waveforms)
        time_loss = F.mse_loss(pred_waveform, target_waveform)

        # 2. Frequency domain loss (mean squared error between FFT magnitudes)
        # Compute the real-valued fast Fourier transform (FFT) of the predicted waveform
        pred_fft = torch.fft.rfft(pred_waveform, dim=-1)
        # Compute the real-valued FFT of the target waveform
        target_fft = torch.fft.rfft(target_waveform, dim=-1)

        # Compute the magnitude of the FFT for both predicted and target waveforms
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Compute the frequency domain loss as the MSE between the FFT magnitudes
        freq_loss = F.mse_loss(pred_mag, target_mag)

        # 3. Dynamically adjust alpha based on the ratio of frequency loss to time loss
        # Compute the loss ratio (frequency loss / time loss), adding a small value (1e-6) to avoid division by zero
        loss_ratio = freq_loss.item() / (time_loss.item() + 1e-6)

        # Update alpha using exponential smoothing:
        # - (1 - smoothing) * self.alpha: Keeps a portion of the previous alpha value
        # - smoothing * loss_ratio: Adds a portion of the new loss ratio
        self.alpha = (1 - self.smoothing) * self.alpha + self.smoothing * loss_ratio

        # 4. Compute the final combined loss
        # - Weight the time loss by (1 - alpha)
        # - Weight the frequency loss by alpha
        total_loss = time_loss * (1 - self.alpha) + freq_loss * self.alpha
    #se non migliora rimuovi (1 -sel.alpha) e rimetti solo self.alpha
        return total_loss  # Return the combined loss

