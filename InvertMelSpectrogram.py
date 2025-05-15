import torchaudio
import torch
# Inverter class to turn Mel spectrogram back to waveform
class MelToWaveform(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, n_mels=80, hop_length=256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length

        self.mel_to_spec = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            n_iter=32
        )

    def forward(self, mel_spec):
        # mel_spec: [batch_size, n_mels, time]
        spec = self.mel_to_spec(mel_spec)
        waveform = self.griffin_lim(spec)
        return waveform
